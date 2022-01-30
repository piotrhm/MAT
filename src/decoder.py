import torch.nn as nn

from src.utils import ScaleNorm, LayerNorm, clones, SublayerConnection


class Decoder(nn.Module):
    def __init__(self, layer, N, scale_norm):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = ScaleNorm(layer.size) if scale_norm else LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, graph):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, graph.adj_matrix, graph.distances_matrix, graph.edges_att)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, scale_norm):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, scale_norm), 3)

    def forward(self, x, memory, src_mask, tgt_mask, adj_matrix, distances_matrix, edges_att):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, adj_matrix, distances_matrix, edges_att, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
