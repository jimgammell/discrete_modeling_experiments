from typing import *
import torch
from torch import nn, optim
import lightning as L

from ..utils import *
from utils import lr_schedulers
import models

class Module(L.LightningModule):
    def __init__(self,
        classifier_name: str,
        input_shape: Sequence[int],
        output_classes: int,
        classifier_kwargs: dict = {},
        lr_scheduler_name: str = None,
        lr_scheduler_kwargs: dict = {},
        lr: float = 2e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        ternary_weight_decay: float = 1e-12,
        full_precision_weight_decay: float = 1.e-4,
        compile_model: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.classifier = models.load(self.hparams.classifier_name, input_shape, output_classes, **self.hparams.classifier_kwargs)
        if compile_model:
            self.classifier = torch.compile(self.classifier)
    
    def configure_optimizers(self):
        ternary_weight_decay, full_precision_weight_decay, no_weight_decay = [], [], []
        for name, param in self.classifier.named_parameters():
            if 'weight_logits' in name:
                ternary_weight_decay.append(param)
            elif ('weight' in name) and not('norm' in name):
                full_precision_weight_decay.append(param)
            else:
                no_weight_decay.append(param)
        param_groups = [
            {'params': ternary_weight_decay, 'weight_decay': self.hparams.ternary_weight_decay},
            {'params': full_precision_weight_decay, 'weight_decay': self.hparams.full_precision_weight_decay},
            {'params': no_weight_decay, 'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(param_groups, lr=self.hparams.lr, betas=(self.hparams.beta_1, self.hparams.beta_2), eps=self.hparams.eps)
        rv = {'optimizer': self.optimizer}
        if self.hparams.lr_scheduler_name is not None:
            if self.trainer.max_epochs != -1:
                total_steps = self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader())
            elif self.trainer.max_steps != -1:
                total_steps = self.trainer.max_steps
            else:
                assert False
            lr_scheduler_constructor = getattr(lr_schedulers, self.hparams.lr_scheduler_name)
            self.lr_scheduler = lr_scheduler_constructor(self.optimizer, total_steps, **self.hparams.lr_scheduler_kwargs)
            rv.update({'lr_scheduler': self.lr_scheduler})
        return rv
    
    def unpack_batch(self, batch):
        x, y = batch
        assert x.size(0) == y.size(0)
        return x, y
    
    def step(self, batch: Tuple[torch.Tensor], train: bool = False):
        if train:
            optimizer = self.optimizers()
            optimizer.zero_grad()
            lr_scheduler = self.lr_schedulers()
        x, y = self.unpack_batch(batch)
        batch_size = x.size(0)
        rv = {}
        logits = self.classifier(x)
        loss = nn.functional.cross_entropy(logits, y)
        rv.update({'loss': loss.detach(), 'acc': (logits.detach().argmax(dim=-1) == y).sum()/batch_size})
        if train:
            self.manual_backward(loss)
            #nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            rv.update({'rms_grad': get_rms_grad(self.classifier)})
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
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