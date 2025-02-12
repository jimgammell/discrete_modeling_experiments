import numpy as np
import torch
from torch import nn

class SparseAttention(nn.Module):
    def __init__(self, hidden_dim: int, head_count: int, noisy_soft: bool = False, eps: float = 1e-4, grad_lpf_beta: float = 0.999):
        super().__init__()
        
        assert hidden_dim % head_count == 0
        self.hidden_dim = hidden_dim
        self.head_count = head_count
        self.noisy_soft = noisy_soft
        self.eps = eps
        self.grad_lpf_beta = grad_lpf_beta
        self.head_dim = self.hidden_dim // self.head_count
        self.scale = self.head_dim**0.5
        
        self.to_qkv = nn.Linear(self.hidden_dim, 3*self.hidden_dim, bias=False)
        self.to_out = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.taut = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.etat = nn.Parameter(torch.tensor(0.0), requires_grad=True)
    
    def get_tau(self):
        return self.eps + self.taut.exp()
    
    def get_eta(self):
        return self.eps + 2*nn.functional.sigmoid(self.etat)
    
    def get_b(self, log_probs):
        u = torch.rand_like(log_probs)
        z = log_probs - (-u.log()).log()
        hard_b = nn.functional.one_hot(z.argmax(dim=-1), num_classes=z.size(-1)).to(log_probs.dtype)
        
    
    def x_to_qkv(self, x):
        batch_size, token_count, dim = x.shape
        qkv = self.to_qkv(x).reshape(batch_size, token_count, self.head_count, 3*self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        return q, k, v
    
    def forward(self, x):
        pass
    
    def __repr__(self):
        pass