import torch
from torch import nn

class RebarModule(nn.Module):
    def __init__(self, trainable_params: bool = False, init_taut: float = 0.0, init_etat: float = 0.0, eps: float = 1e-6):
        super().__init__()
        
        self.trainable_params = trainable_params
        self.rebar_eps = eps
        self.init_taut = init_taut
        self.init_etat = init_etat
        if self.trainable_params:
            self.rebar_taut = nn.Parameter(torch.tensor(self.init_taut, dtype=torch.float32), requires_grad=True)
            self.rebar_etat = nn.Parameter(torch.tensor(self.init_etat, dtype=torch.float32), requires_grad=True)
        else:
            self.register_buffer('rebar_taut', torch.tensor(self.init_taut, dtype=torch.float32))
            self.register_buffer('rebar_etat', torch.tensor(self.init_etat, dtype=torch.float32))
    
    @torch.no_grad()
    def rand(self, **kwargs):
        return self.rebar_eps + (1-2*self.rebar_eps)*torch.rand(**kwargs)
    
    @torch.no_grad()
    def rand_like(self, x, **kwargs):
        return self.rebar_eps + (1-2*self.rebar_eps)*torch.rand_like(x, **kwargs)
    
    def get_rebar_tau(self):
        return self.rebar_eps + self.rebar_taut.exp()
    
    def get_rebar_eta(self):
        return self.rebar_eps + 2*nn.functional.sigmoid(self.rebar_etat)