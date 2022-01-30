import torch.nn as nn

from src.utils import ScaleNorm, LayerNorm, clones, SublayerConnection
from src.types import Graph


class Encoder(nn.Module):
    def __init__(self, layer, N, scale_norm):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = ScaleNorm(layer.size) if scale_norm else LayerNorm(layer.size)

    def forward(self, x, mask, graph: Graph):
        for layer in self.layers:
            x = layer(x, mask, graph.adj_matrix, graph.distances_matrix, graph.edges_att)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, scale_norm):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, scale_norm), 2)
        self.size = size

    def forward(self, x, mask, adj_matrix, distances_matrix, edges_att):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, adj_matrix, distances_matrix, edges_att, mask))
        return self.sublayer[1](x, self.feed_forward)