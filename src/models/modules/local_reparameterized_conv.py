from typing import *
import numpy as np
import torch
from torch import nn

class TernaryConv2d(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        l2_penalty_coeff: float = 0.0,
        scale_and_shift: bool = True,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(2*[kernel_size]) if isinstance(kernel_size, int) else kernel_size
        self.stride = tuple(2*[stride]) if isinstance(stride, int) else stride
        self.padding = tuple(2*[padding]) if isinstance(padding, int) else padding
        self.l2_penalty_coeff = l2_penalty_coeff
        self.scale_and_shift = scale_and_shift
        self.dilation = tuple(2*[dilation]) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.device = device
        self.dtype = dtype
        
        self.weight_logits = nn.Parameter(torch.full((self.out_channels, self.in_channels//self.groups, *self.kernel_size, 2), torch.nan, device=self.device, dtype=self.dtype), requires_grad=True)
        if self.scale_and_shift:
            self.scale = nn.Parameter(torch.full((self.out_channels,), torch.nan, dtype=torch.float32), requires_grad=True)
            self.shift = nn.Parameter(torch.full((self.out_channels,), torch.nan, dtype=torch.float32), requires_grad=True)
        self.register_buffer('eval_weight', torch.full_like(self.weight_logits[..., 0], np.nan))
        self.stale_eval_params = True
        
        fan_in = nn.init._calculate_correct_fan(self.weight_logits[..., 0], 'fan_in')
        gain = nn.init.calculate_gain('relu')
        std = gain / np.sqrt(fan_in)
        bound = np.sqrt(3.) * std
        nn.init.uniform_(self.weight_logits[..., 0], a=np.log(0.5)-np.log1p(-0.5)-0.01, b=np.log(0.5)-np.log1p(-0.5)+0.01)
        nn.init.uniform_(self.weight_logits[..., 1], a=-0.01, b=0.01)
        if self.scale_and_shift:
            nn.init.constant_(self.scale, bound)
            nn.init.constant_(self.shift, 0.01)
        self.refresh_eval_params()
    
    def refresh_eval_params(self):
        if self.stale_eval_params:
            weight_mag = (self.weight_logits[..., 0] > 0).to(torch.float32)
            weight_sign = 2*(self.weight_logits[..., 1] > 0).to(torch.float32) - 1
            self.eval_weight = weight_sign*weight_mag
            self.stale_eval_params = False
    
    @torch.compile()
    def train_forward(self, x):
        self.stale_eval_params = True
        weight_p = nn.functional.sigmoid(self.weight_logits)
        weight_p_mag = weight_p[..., 0]
        weight_p_sgn = weight_p[..., 1]
        weight_mean = 2*weight_p_mag*weight_p_sgn - weight_p_mag
        weight_var = weight_p_mag - weight_p_mag.pow(2)*(2*weight_p_sgn - 1).pow(2)
        out_mean = nn.functional.conv2d(
            x, weight_mean, None, self.stride, self.padding, self.dilation, self.groups
        )
        out_var = nn.functional.conv2d(
            x.pow(2), weight_var, None, self.stride, self.padding, self.dilation, self.groups
        )
        out = out_mean + out_var.sqrt()*torch.randn_like(out_var)
        if self.scale_and_shift:
            out = out*self.scale.reshape(1, self.out_channels, 1, 1) + self.shift.reshape(1, self.out_channels, 1, 1)
        if self.l2_penalty_coeff > 0:
            l2_penalty = self.l2_penalty_coeff*self.weight_logits.pow(2).sum()
            out = out + l2_penalty - l2_penalty.detach()
        return out
    
    @torch.compile()
    def eval_forward(self, x):
        self.refresh_eval_params()
        out = nn.functional.conv2d(
            x, self.eval_weight, None, self.stride, self.padding, self.dilation, self.groups
        )
        if self.scale_and_shift:
            out = out*self.scale.reshape(1, self.out_channels, 1, 1) + self.shift.reshape(1, self.out_channels, 1, 1)
        return out
    
    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size[0] if all(x == self.kernel_size[0] for x in self.kernel_size) else self.kernel_size})'
        )