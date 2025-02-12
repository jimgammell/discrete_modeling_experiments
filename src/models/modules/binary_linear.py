from typing import *
import numpy as np
import torch
from torch import nn

from .rebar_module import RebarModule

class BinaryLinear(RebarModule):
    def __init__(self, in_dims: int, out_dims: int, bias=True):
        super().__init__()
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.bias = bias
        if self.bias:
            self.in_dims += 1
        
        self.weight_logits = nn.Parameter(torch.empty(self.out_dims, self.in_dims, dtype=torch.float32), requires_grad=True)
        self.scalar = nn.Parameter(torch.tensor(np.log(np.exp(1)-1), dtype=torch.float32), requires_grad=True)
        
        # Initialize weights to something analogous to nn.init.xavier_uniform_, based on assumption that tanh(x) == x for small x.
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight_logits[:, :-1])
        gain = nn.init.calculate_gain('relu')
        a_xavier = gain*np.sqrt(6./(fan_in + fan_out))
        a = np.log1p(a_xavier) - np.log1p(-a_xavier)
        nn.init.uniform_(self.weight_logits[:, :-1], a=-a, b=a)
        nn.init.constant_(self.weight_logits[:, -1], 0)
        
    def get_weights(self, batch_size: int, temperature: Optional[torch.Tensor] = None):
        log_probs = nn.functional.logsigmoid(self.weight_logits).reshape(1, self.in_dims, self.out_dims).expand(batch_size, self.in_dims, self.out_dims)
        log_1mprobs = nn.functional.logsigmoid(-self.weight_logits).reshape(1, self.in_dims, self.out_dims).expand(batch_size, self.in_dims, self.out_dims)
        log_alpha = log_probs - log_1mprobs
        u = self.rand_like(log_probs)
        binary_weights = torch.where(log_alpha + u.log() - (1-u).log() >= 0, torch.ones_like(u), torch.zeros_like(u))
        if self.training:
            log_prob_binary_weights = (binary_weights*log_probs + (1-binary_weights)*log_1mprobs).reshape(batch_size, -1).sum(dim=-1)
            uprime = 1 - log_probs.exp()
            v = self.rand_like(u)
            v = torch.where(binary_weights==1, uprime + v*(1-uprime), v*uprime).clip_(self.rebar_eps, 1-self.rebar_eps)
            to_z = lambda log_alpha, u: ((temperature**2 + temperature + 1)/(temperature + 1))*log_alpha + u.log() - (1-u).log()
            relaxed_weights = nn.functional.sigmoid(to_z(log_alpha, u)/temperature)
            relaxed_weights_tilde = nn.functional.sigmoid(to_z(log_alpha, v)/temperature)
            relaxed_weights_tilde_detached = nn.functional.sigmoid(to_z(log_alpha.detach(), v.detach())/temperature)
            return (binary_weights, relaxed_weights, relaxed_weights_tilde, relaxed_weights_tilde_detached), log_prob_binary_weights
        else:
            return binary_weights
    
    def forward(self, x):
        batch_size, *dims = x.shape
        if self.bias:
            x = torch.cat([x, torch.ones_like(x[..., 0, None])], dim=-1)
        if self.training:
            tau = self.get_rebar_tau()
            eta = self.get_rebar_eta()
            weights, log_probs = self.get_weights(batch_size, temperature=tau)
            binary_out, relaxed_out, relaxed_out_tilde, relaxed_out_tilde_detached = map(
                lambda weight: torch.bmm(x.unsqueeze(-2), 2.*weight - 1.).squeeze(-2), weights
            )
            rebar_out = (binary_out - eta*relaxed_out_tilde_detached)*log_probs.unsqueeze(-1) + eta*relaxed_out - eta*relaxed_out_tilde
            out = binary_out.detach() + rebar_out - rebar_out.detach()
            out = out * nn.functional.softplus(self.scalar)
        else:
            weights = self.get_weights(batch_size)
            out = torch.bmm(x.unsqueeze(-2), 2.*weights - 1).squeeze(-2)
        return out
    
    def __repr__(self):
        return f'{self.__class__.__name__}(in_dims={self.in_dims-1}, out_dims={self.out_dims}, bias={self.bias})'