import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import xavier_normal_small_init_, xavier_uniform_small_init_, clones
from src.attention import MultiHeadedAttentionMAT, MultiHeadedAttention
from src.generator import PositionGenerator
from src.encoder import Encoder, EncoderLayer
from src.decoder import Decoder, DecoderLayer
from src.types import Graph

from src.gumbel import gumbel_softmax


def make_model(d_atom, N=2, d_model=128, h=8, dropout=0.1, 
               lambda_attention=0.3, lambda_distance=0.3, trainable_lambda=False,
               N_dense=2, leaky_relu_slope=0.0, aggregation_type='mean', 
               dense_output_nonlinearity='relu', distance_matrix_kernel='softmax',
               use_edge_features=False, n_output=1,
               control_edges=False, integrated_distances=False, 
               scale_norm=False, init_type='uniform', n_generator_layers=1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn_mat = MultiHeadedAttentionMAT(h, d_model, dropout, lambda_attention, lambda_distance, trainable_lambda,
                                   distance_matrix_kernel, use_edge_features, control_edges, integrated_distances)
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, N_dense, dropout, leaky_relu_slope, dense_output_nonlinearity)
    model = GraphTransformer(
        Encoder(EncoderLayer(d_model, c(attn_mat), c(ff), dropout, scale_norm), N, scale_norm),
        Decoder(DecoderLayer(d_model, c(attn_mat), c(attn), c(ff), dropout, scale_norm), N, scale_norm),
        Embeddings(d_model, d_atom, dropout),
        Embeddings(d_model, d_atom, dropout),
        PositionGenerator(d_model, d_atom),
        d_model,
    )
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == 'uniform':
                nn.init.xavier_uniform_(p)
            elif init_type == 'normal':
                nn.init.xavier_normal_(p)
            elif init_type == 'small_normal_init':
                xavier_normal_small_init_(p)
            elif init_type == 'small_uniform_init':
                xavier_uniform_small_init_(p)
    return model


class GraphTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, d_model):
        super(GraphTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.d_model = d_model

    def forward_encoder(self, src, src_mask, graph: Graph):
        return self.predict_property(self.encode(src, src_mask, graph), src_mask)

    def forward(self, src, tgt, src_mask, tgt_mask, graph: Graph, gumbel=False, temp=1.0, hard=False):
        tgt_graph = copy.deepcopy(graph)

        if gumbel:
            q = self.encode(src, src_mask, graph)
            q_y = q.view(q.size(0), -1, self.d_model)
            z = gumbel_softmax(q_y, temp, hard)
            z_y = z.view(q.size(0), -1, self.d_model)
            out = self.decode(z_y, None, tgt, tgt_mask, tgt_graph)
            return self.generator(out, src_mask), F.softmax(q, dim=-1).reshape(*q.size())

        # TODO fix src_mask, maybe unnecessary?
        out = self.decode(self.encode(src, src_mask, graph), None, tgt, tgt_mask, tgt_graph)
        return self.generator(out, src_mask)
    
    def encode(self, src, src_mask, graph: Graph):
        return self.encoder(self.src_embed(src), src_mask, graph)

    def decode(self, memory, src_mask, tgt, tgt_mask, graph: Graph):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, graph)
    
    def predict_property(self, out, out_mask):
        return self.generator(out, out_mask)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.0, dense_output_nonlinearity='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x

    def forward(self, x):
        if self.N_dense == 0:
            return x
        
        for i in range(len(self.linears)-1):
            x = self.dropout[i](F.leaky_relu(self.linears[i](x), negative_slope=self.leaky_relu_slope))
            
        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))


class Embeddings(nn.Module):
    def __init__(self, d_model, d_atom, dropout):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(d_atom, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))
