from typing import *
import numpy as np
import torch
from torch import nn

from ._base_module import _TernaryModule

class TernaryConv2d(_TernaryModule):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        scale_and_shift: bool = False,
        initial_weights: Optional[torch.Tensor] = None,
        p_min: float = 0.05,
        p_max: float = 0.95,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = 'cpu'
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(2*[kernel_size]) if isinstance(kernel_size, int) else kernel_size
        self.stride = tuple(2*[stride]) if isinstance(stride, int) else stride
        self.padding = tuple(2*[padding]) if isinstance(padding, int) else padding
        self.dilation = tuple(2*[dilation]) if isinstance(dilation, int) else dilation
        self.groups = groups
        if initial_weights is None:
            initial_weights = torch.full((self.out_channels, self.in_channels//self.groups, *self.kernel_size), torch.nan, device=device, dtype=dtype)
            nn.init.kaiming_uniform_(initial_weights, a=np.sqrt(5))
        super().__init__(
            weight_dims=(self.out_channels, self.in_channels//self.groups, *self.kernel_size),
            linear_fn=self.linear_fn,
            initial_weights=initial_weights,
            scale_and_shift_dims=(1, self.out_channels, 1, 1) if scale_and_shift else None,
            p_min=p_min,
            p_max=p_max,
            logits_dtype=dtype,
            device=device
        )
    
    def linear_fn(self, input, weight):
        return nn.functional.conv2d(input, weight, None, self.stride, self.padding, self.dilation, self.groups)

    def __repr__(self):
        properties = []
        properties.append(f'in_channels={self.in_channels}')
        properties.append(f'out_channels={self.out_channels}')
        properties.append(f'kernel_size={self.kernel_size[0] if all(x == self.kernel_size[0] for x in self.kernel_size) else self.kernel_size}')
        if self.stride != (1, 1):
            properties.append(f'stride={self.stride[0] if all(x == self.stride[0] for x in self.stride) else self.stride}')
        if self.padding != (0, 0):
            properties.append(f'padding={self.padding[0] if all(x == self.padding[0] for x in self.padding) else self.padding}')
        if self.dilation != (1, 1):
            properties.append(f'dilation={self.dilation[0] if all(x == self.dilation[0] for x in self.dilation) else self.dilation}')
        if self.groups != 1:
            properties.append(f'groups={self.groups}')
        properties.append(f'scale_and_shift={self.scale_and_shift}')
        return f'{self.__class__.__name__}({", ".join(properties)})'