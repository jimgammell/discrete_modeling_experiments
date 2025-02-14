from typing import *
import numpy as np
import torch
from torch import nn

from ._base_module import _TernaryModule

class TernaryLinear(_TernaryModule):
    def __init__(self,
        in_dims: int,
        out_dims: int,
        p_min: float = 0.05,
        p_max: float = 0.95,
        scale_and_shift: bool = False,
        initial_weights: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = 'cpu'
    ):
        self.in_dims = in_dims
        self.out_dims = out_dims
        if initial_weights is None:
            initial_weights = torch.full((out_dims, in_dims), torch.nan, device=device, dtype=dtype)
            nn.init.kaiming_uniform_(initial_weights, a=np.sqrt(5))
        super().__init__(
            weight_dims=(out_dims, in_dims),
            linear_fn=self.linear_fn,
            initial_weights=initial_weights,
            scale_and_shift_dims=(1, self.out_dims) if scale_and_shift else None,
            p_min=p_min,
            p_max=p_max,
            logits_dtype=dtype,
            device=device
        )
    
    def linear_fn(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, weight, bias=None)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(in_dims={self.in_dims}, out_dims={self.out_dims}, scale_and_shift={self.scale_and_shift})'