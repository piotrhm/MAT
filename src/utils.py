import math
import copy

import torch
import torch.nn as nn

from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_, _no_grad_uniform_


def earily_stop(val_acc_history, tasks, early_stop_step_single, early_stop_step_multi, required_progress):
    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_acc_history: a list contains all the historical validation acc
    @param required_progress: the next acc should be higher than the previous by
        at least required_progress amount to be non-trivial
    @param t: number of training steps
    @return: a boolean indicates if the model should earily stop
    """
    # TODO: add your code here
    if len(tasks) == 1:
        t = early_stop_step_single
    else:
        t = early_stop_step_multi

    if len(val_acc_history)>t:
        if val_acc_history[-1] - val_acc_history[-1-t] < required_progress:
            return True
    return False


def xavier_normal_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4*fan_out))

    return _no_grad_normal_(tensor, 0., std)


def xavier_uniform_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4*fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    "All g’s in SCALE NORM are initialized to sqrt(d)"

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, scale_norm):
        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])