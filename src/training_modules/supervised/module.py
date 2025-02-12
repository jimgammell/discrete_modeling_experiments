from typing import *
import torch
from torch import nn, optim
import lightning as L

from ..utils import *
import models

class Module(L.LightningModule):
    def __init__(self,
        classifier_name: str,
        classifier_kwargs: dict = {},
        lr: float = 2e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.classifier = models.load(self.hparams.classifier_name, **self.hparams.classifier_kwargs)
    
    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta_1, self.hparams.beta_2), eps=self.hparams.eps)
        return {'optimizer': self.optimizer}
    
    def unpack_batch(self, batch):
        x, y = batch
        assert x.size(0) == y.size(0)
        return x, y
    
    def step(self, batch: Tuple[torch.Tensor], train: bool = False):
        if train:
            optimizer = self.optimizers()
            optimizer.zero_grad()
        x, y = self.unpack_batch(batch)
        batch_size = x.size(0)
        rv = {}
        logits = self.classifier(x)
        loss = nn.functional.cross_entropy(logits, y)
        rv.update({'loss': loss.detach(), 'acc': (logits.detach().argmax(dim=-1) == y).sum()/batch_size})
        if train:
            self.manual_backward(loss)
            rv.update({'rms_grad': get_rms_grad(self.classifier)})
            optimizer.step()
        assert all(torch.all(torch.isfinite(param)) for param in self.classifier.parameters())
        return rv
    
    def training_step(self, batch: Tuple[torch.Tensor]):
        rv = self.step(batch, train=True)
        for key, val in rv.items():
            self.log(f'train_{key}', val, on_step=False, on_epoch=True)
    
    def validation_step(self, batch: Tuple[torch.Tensor]):
        rv = self.step(batch, train=False)
        for key, val in rv.items():
            self.log(f'val_{key}', val, on_step=False, on_epoch=True)