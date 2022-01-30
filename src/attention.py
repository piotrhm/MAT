import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import clones


def attention_mat(query, key, value, adj_matrix, distances_matrix, edges_att,
                  mask=None, dropout=None,
                  lambdas=(0.3, 0.3, 0.4), trainable_lambda=False,
                  distance_matrix_kernel=None, use_edge_features=False, control_edges=False,
                  eps=1e-6, inf=1e12):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1) == 0, -inf)
    p_attn = F.softmax(scores, dim=-1)

    if use_edge_features:
        adj_matrix = edges_att.view(adj_matrix.shape)

    # Prepare adjacency matrix
    adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1).unsqueeze(2) + eps)
    adj_matrix = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
    p_adj = adj_matrix

    p_dist = distances_matrix

    if trainable_lambda:
        softmax_attention, softmax_distance, softmax_adjacency = lambdas.cuda()
        p_weighted = softmax_attention * p_attn + softmax_distance * p_dist + softmax_adjacency * p_adj
    else:
        lambda_attention, lambda_distance, lambda_adjacency = lambdas
        p_weighted = lambda_attention * p_attn + lambda_distance * p_dist + lambda_adjacency * p_adj

    if dropout is not None:
        p_weighted = dropout(p_weighted)

    atoms_featrues = torch.matmul(p_weighted, value)
    return atoms_featrues, p_weighted, p_attn


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class EdgeFeaturesLayer(nn.Module):
    def __init__(self, d_model, d_edge, h, dropout):
        super(EdgeFeaturesLayer, self).__init__()
        assert d_model % h == 0
        d_k = d_model // h
        self.linear = nn.Linear(d_edge, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(0.25)

    def forward(self, x):
        p_edge = x.permute(0, 2, 3, 1)
        p_edge = self.linear(p_edge).permute(0, 3, 1, 2)
        return torch.relu(p_edge)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiHeadedAttentionMAT(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, lambda_attention=0.3, lambda_distance=0.3, trainable_lambda=False,
                 distance_matrix_kernel='softmax', use_edge_features=False, control_edges=False,
                 integrated_distances=False):
        super(MultiHeadedAttentionMAT, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.trainable_lambda = trainable_lambda
        if trainable_lambda:
            lambda_adjacency = 1. - lambda_attention - lambda_distance
            lambdas_tensor = torch.tensor([lambda_attention, lambda_distance, lambda_adjacency], requires_grad=True)
            self.lambdas = torch.nn.Parameter(lambdas_tensor)
        else:
            lambda_adjacency = 1. - lambda_attention - lambda_distance
            self.lambdas = (lambda_attention, lambda_distance, lambda_adjacency)

        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        if distance_matrix_kernel == 'softmax':
            self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
        elif distance_matrix_kernel == 'exp':
            self.distance_matrix_kernel = lambda x: torch.exp(-x)
        self.integrated_distances = integrated_distances
        self.use_edge_features = use_edge_features
        self.control_edges = control_edges
        if use_edge_features:
            d_edge = 11 if not integrated_distances else 12
            self.edges_feature_layer = EdgeFeaturesLayer(d_model, d_edge, h, dropout)

    def forward(self, query, key, value, adj_matrix, distances_matrix, edges_att, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # Prepare distances matrix
        distances_matrix = distances_matrix.masked_fill(mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)
        distances_matrix = self.distance_matrix_kernel(distances_matrix)
        p_dist = distances_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

        if self.use_edge_features:
            if self.integrated_distances:
                edges_att = torch.cat((edges_att, distances_matrix.unsqueeze(1)), dim=1)
            edges_att = self.edges_feature_layer(edges_att)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn, self.self_attn = attention_mat(query, key, value, adj_matrix,
                                                     p_dist, edges_att,
                                                     mask=mask, dropout=self.dropout,
                                                     lambdas=self.lambdas,
                                                     trainable_lambda=self.trainable_lambda,
                                                     distance_matrix_kernel=self.distance_matrix_kernel,
                                                     use_edge_features=self.use_edge_features,
                                                     control_edges=self.control_edges)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
