from typing import *
import numpy as np
import torch
from torch import nn

class _TernaryModule(nn.Module):
    def __init__(self,
        weight_dims: Sequence[int],
        linear_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        initial_weights: torch.Tensor,
        scale_and_shift_dims: Optional[Sequence[int]] = None,
        p_min: float = 0.05,
        p_max: float = 0.95,
        logits_dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = 'cpu'
    ):
        super().__init__()
        self.weight_dims = weight_dims
        self.linear_fn = linear_fn
        self.initial_weights = initial_weights/initial_weights.std()
        assert all(d1 == d2 for d1, d2 in zip(self.weight_dims, self.initial_weights.shape))
        self.scale_and_shift_dims = scale_and_shift_dims
        self.scale_and_shift = self.scale_and_shift_dims is not None
        self.p_min = p_min
        self.p_max = p_max
        self.logits_dtype = logits_dtype
        self.device = device
        self.construct()
        
    def construct(self):
        self.weight_logits = nn.Parameter(torch.full((*self.weight_dims, 2), torch.nan, dtype=self.logits_dtype, device=self.device), requires_grad=True)
        if self.scale_and_shift:
            self.scale = nn.Parameter(torch.full(self.scale_and_shift_dims, torch.nan, dtype=self.logits_dtype, device=self.device), requires_grad=True)
            self.shift = nn.Parameter(torch.full(self.scale_and_shift_dims, torch.nan, dtype=self.logits_dtype, device=self.device), requires_grad=True)
        self.register_buffer('eval_weight', torch.full(self.weight_dims, torch.nan, dtype=self.logits_dtype, device=self.device))
        self.stale_eval_params = True
        
        p_0 = self.p_max - (self.p_max - self.p_min)*self.initial_weights.abs()
        p_1 = 0.5*(1 + self.initial_weights/(1 - p_0))
        p_0 = p_0.clamp(self.p_min, self.p_max)
        p_1 = p_1.clamp(self.p_min, self.p_max)
        self.weight_logits.data[..., 0] = np.log(p_0) - np.log1p(-p_0)
        self.weight_logits.data[..., 1] = np.log(p_1) - np.log1p(-p_1)
        if self.scale_and_shift:
            nn.init.constant_(self.scale, 1)
            nn.init.constant_(self.shift, 0.01)
        self.refresh_eval_params()
    
    def refresh_eval_params(self):
        if self.stale_eval_params:
            weight_mag = (self.weight_logits[..., 0] > 0).to(torch.float32)
            weight_sgn = 2*(self.weight_logits[..., 1] > 0).to(torch.float32) - 1
            self.eval_weight = weight_sgn*weight_mag
            self.stale_eval_params = False
    
    def apply_scale_and_shift(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_and_shift:
            x = nn.functional.softplus(self.scale)*x + self.shift
        return x
    
    def train_forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_p = nn.functional.sigmoid(self.weight_logits)
        weight_p_mag = weight_p[..., 0]
        weight_p_sgn = weight_p[..., 1]
        weight_mean = 2*weight_p_mag*weight_p_sgn - weight_p_mag
        weight_var = weight_p_mag - weight_p_mag.pow(2)*(2*weight_p_sgn - 1).pow(2)
        out_mean = self.linear_fn(x, weight_mean)
        out_var = self.linear_fn(x.pow(2), weight_var)
        out = out_mean + out_var.sqrt()*torch.randn_like(out_var)
        out = self.apply_scale_and_shift(out)
        return out
    
    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.refresh_eval_params()
        out = self.linear_fn(x, self.eval_weight)
        out = self.apply_scale_and_shift(out)
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            out = self.train_forward(x)
        else:
            out = self.eval_forward(x)
        if out.requires_grad:
            out.register_hook(self._backward_hook)
        return out
    
    def _backward_hook(self, grad: Any):
        assert self.training
        self.stale_eval_params = True