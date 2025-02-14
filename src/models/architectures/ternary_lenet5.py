from typing import *
from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from ..local_reparameterized_modules import TernaryLinear, TernaryConv2d

class TernaryLeNet5(nn.Module):
    def __init__(self, input_shape: Sequence[int], output_classes: int, base_channels: int = 32, dense_width: int = 128):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.base_channels = base_channels
        self.dense_width = dense_width
        
        self.conv_stage = nn.Sequential(OrderedDict([
            ('conv1', TernaryConv2d(self.input_shape[0], self.base_channels, kernel_size=5, stride=1, padding=2, scale_and_shift=False)),
            ('norm1', nn.BatchNorm2d(self.base_channels)),
            ('act1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2)),
            ('conv2', TernaryConv2d(self.base_channels, 2*self.base_channels, kernel_size=5, stride=1, padding=2, scale_and_shift=False)),
            ('norm2', nn.BatchNorm2d(2*self.base_channels)),
            ('act2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2))
        ]))
        self.dense_stage = nn.Sequential(OrderedDict([
            ('dense1', TernaryLinear(2*self.base_channels*(self.input_shape[1]//4)*(self.input_shape[2]//4), self.dense_width, scale_and_shift=True)),
            ('dropout', nn.Dropout(0.5)),
            ('act1', nn.ReLU()),
            ('dense2', nn.Linear(self.dense_width, self.output_classes, bias=False))
        ]))
        nn.init.xavier_uniform_(self.dense_stage.dense2.weight)
    
    def forward(self, x):
        batch_size, *dims = x.shape
        x = self.conv_stage(x)
        x = self.dense_stage(x.reshape(batch_size, -1))
        return x