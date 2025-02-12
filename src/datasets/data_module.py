from typing import *
import os
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
import lightning as L

class DataModule(L.LightningDataModule):
    def __init__(self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        val_prop: float = 0.2,
        train_batch_size: Optional[int] = 256,
        eval_batch_size: Optional[int] = 256,
        dataloader_kwargs: dict = {}
    ):
        super().__init__()
        self.base_train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_prop = val_prop
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        default_dataloader_kwargs = dict(
            num_workers=max(os.cpu_count()//4, 1),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        default_dataloader_kwargs.update(dataloader_kwargs)
        self.dataloader_kwargs = default_dataloader_kwargs
        self.train_indices = self.val_indices = None
    
    def setup(self, stage: str):
        stage = None
        self.val_length = int(self.val_prop*len(self.base_train_dataset))
        self.train_length = len(self.base_train_dataset) - self.val_length
        if (self.train_indices is None) or (self.val_indices is None):
            indices = np.random.choice(len(self.base_train_dataset), len(self.base_train_dataset), replace=False)
            self.train_indices = indices[:self.train_length]
            self.val_indices = indices[self.train_length:]
        self.train_dataset = Subset(self.base_train_dataset, self.train_indices)
        self.val_dataset = Subset(self.base_train_dataset, self.val_indices)
        if self.train_batch_size is None:
            self.train_batch_size = len(self.train_dataset)
        if self.eval_batch_size is None:
            self.eval_batch_size = len(self.val_dataset)
    
    def train_dataloader(self, override_batch_size: Optional[int] = None):
        batch_size = min(self.train_batch_size if override_batch_size is None else override_batch_size, len(self.train_dataset))
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **self.dataloader_kwargs)
        return train_dataloader
    
    def val_dataloader(self, override_batch_size: Optional[int] = None):
        batch_size = min(self.eval_batch_size if override_batch_size is None else override_batch_size, len(self.val_dataset))
        val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, **self.dataloader_kwargs)
        return val_dataloader
    
    def test_dataloader(self, override_batch_size: Optional[int] = None):
        batch_size = min(self.eval_batch_size if override_batch_size is None else override_batch_size, len(self.test_dataset))
        test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, **self.dataloader_kwargs)
        return test_dataloader
    
    def on_save_checkpoint(self):
        return dict(train_indices=self.train_indices, val_indices=self.val_indices)
    
    def on_load_checkpoint(self, checkpoint):
        self.train_indices = checkpoint.get('train_indices', None)
        self.val_indices = checkpoint.get('val_indices', None)