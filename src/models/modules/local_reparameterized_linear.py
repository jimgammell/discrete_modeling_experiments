import numpy as np
import torch
from torch import nn

class TernaryLinear(nn.Module):
    def __init__(self,
        in_dims: int,
        out_dims: int,
        p_max: float = 0.95,
        l2_penalty_coeff: float = 0.0,
        scale_and_shift: bool = True
    ):
        super().__init__()
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert 0 <= p_max <= 1
        self.p_max = p_max
        self.l2_penalty_coeff = l2_penalty_coeff
        self.scale_and_shift = scale_and_shift
        
        self.weight_logits = nn.Parameter(torch.full((self.out_dims, self.in_dims, 2), torch.nan, dtype=torch.float32), requires_grad=True)
        if self.scale_and_shift:
            self.scale = nn.Parameter(torch.full((self.out_dims,), torch.nan, dtype=torch.float32), requires_grad=True)
            self.shift = nn.Parameter(torch.full((self.out_dims,), torch.nan, dtype=torch.float32), requires_grad=True)
        self.register_buffer('eval_weight', torch.full((self.out_dims, self.in_dims), torch.nan, dtype=torch.float32))
        self.stale_eval_params = True
        
        gain = nn.init.calculate_gain('relu')
        a = gain*np.sqrt(6./(self.in_dims + self.out_dims))
        nn.init.uniform_(self.weight_logits[..., 0], a=np.log(0.5)-np.log1p(-0.5)-a, b=np.log(0.5)-np.log1p(-0.5)+a)
        nn.init.uniform_(self.weight_logits[..., 1], a=-a, b=a)
        if scale_and_shift:
            nn.init.constant_(self.scale, a)
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
        out_mean = nn.functional.linear(x, weight_mean, bias=None)
        out_var = nn.functional.linear(x.pow(2), weight_var, bias=None)
        out = out_mean + out_var.sqrt()*torch.randn_like(out_var)
        if self.scale_and_shift:
            out = out*self.scale.unsqueeze(0) + self.shift.unsqueeze(0)
        if self.l2_penalty_coeff > 0.0:
            l2_penalty = self.l2_penalty_coeff*self.weight_logits.pow(2).sum()
            out = out + l2_penalty - l2_penalty.detach()
        return out
    
    @torch.compile()
    def eval_forward(self, x):
        self.refresh_eval_params()
        out = nn.functional.linear(x, self.eval_weight, bias=None)
        if self.scale_and_shift:
            out = out*self.scale.unsqueeze(0) + self.shift.unsqueeze(0)
        return out
    
    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)
        
    def __repr__(self):
        return f'{self.__class__.__name__}(in_dims={self.in_dims}, out_dims={self.out_dims})'