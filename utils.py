import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        m.weight.data.mul_(1e-3)
        m.bias.data.fill_(0)