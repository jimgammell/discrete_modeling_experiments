from typing import *
from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from ..local_reparameterized_modules import TernaryLinear

class TernaryMLP(nn.Module):
    def __init__(self, input_shape: Sequence[int], output_classes: int, hidden_layer_count: int = 1, hidden_layer_dim: int = 2048):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.hidden_layer_count = hidden_layer_count
        self.hidden_layer_dim = hidden_layer_dim
        
        modules = []
        in_dims = np.prod(self.input_shape)
        out_dims = self.hidden_layer_dim
        for layer_idx in range(self.hidden_layer_count):
            modules.append((f'layer_{layer_idx+1}', TernaryLinear(in_dims, out_dims)))
            modules.append((f'relu_{layer_idx+1}', nn.ReLU()))
            in_dims = out_dims
            out_dims = self.hidden_layer_dim
        modules.append(('output_layer', TernaryLinear(in_dims, self.output_classes)))
        self.model = nn.Sequential(OrderedDict(modules))
    
    def forward(self, x):
        batch_size, *dims = x.shape
        return self.model(x.reshape(batch_size, np.prod(dims)))