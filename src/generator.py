import torch
import torch.nn as nn

from src.utils import ScaleNorm, LayerNorm


class Generator(nn.Module):
    def __init__(self, d_model, aggregation_type='mean', n_output=1, n_layers=1,
                 leaky_relu_slope=0.01, dropout=0.0, scale_norm=False):
        super(Generator, self).__init__()
        if n_layers == 1:
            self.proj = nn.Linear(d_model, n_output)
        else:
            self.proj = []
            for i in range(n_layers - 1):
                self.proj.append(nn.Linear(d_model, d_model))
                self.proj.append(nn.LeakyReLU(leaky_relu_slope))
                self.proj.append(ScaleNorm(d_model) if scale_norm else LayerNorm(d_model))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(d_model, n_output))
            self.proj = torch.nn.Sequential(*self.proj)
        self.aggregation_type = aggregation_type

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = x * mask
        if self.aggregation_type == 'mean':
            out_sum = out_masked.sum(dim=1)
            mask_sum = mask.sum(dim=(1))
            out_avg_pooling = out_sum / mask_sum
        elif self.aggregation_type == 'sum':
            out_sum = out_masked.sum(dim=1)
            out_avg_pooling = out_sum
        elif self.aggregation_type == 'dummy_node':
            out_avg_pooling = out_masked[:, 0]
        projected = self.proj(out_avg_pooling)
        return projected


class PositionGenerator(nn.Module):
    def __init__(self, d_model, d_atom):
        super(PositionGenerator, self).__init__()
        self.norm = LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_atom)

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = self.norm(x) * mask
        projected = self.proj(out_masked)
        return projected
