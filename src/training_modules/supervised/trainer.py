import os
import shutil
from copy import copy
from typing import *
from torch.utils.data import Dataset
from lightning import Trainer as LightningTrainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from ..utils import *
from .module import Module
from .plot_things import plot_training_curves
from datasets.data_module import DataModule

class Trainer:
    def __init__(self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        base_module_kwargs: dict = {},
        base_datamodule_kwargs: dict = {}
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.base_module_kwargs = base_module_kwargs
        self.base_datamodule_kwargs = base_datamodule_kwargs
        self.data_module = DataModule(
            self.train_dataset, self.test_dataset, **self.base_datamodule_kwargs
        )
    
    def run(self,
        save_dir: Union[str, os.PathLike],
        max_epochs: int = 100,
        override_module_kwargs: dict = {}
    ):
        if not training_complete(save_dir):
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)
            kwargs = copy(self.base_module_kwargs)
            kwargs.update(override_module_kwargs)
            training_module = Module(
                input_shape=self.train_dataset.input_shape, output_classes=self.train_dataset.output_classes, **kwargs
            )
            checkpoint = ModelCheckpoint(
                monitor='val_acc', mode='max', save_top_k=1, dirpath=save_dir, filename='best_checkpoint'
            )
            trainer = LightningTrainer(
                max_epochs=max_epochs,
                val_check_interval=1.,
                default_root_dir=save_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(save_dir, name='lightning_output'),
                callbacks=[checkpoint]
            )
            trainer.fit(training_module, datamodule=self.data_module)
            trainer.save_checkpoint(os.path.join(save_dir, 'final_checkpoint.ckpt'))
            extract_training_curves(save_dir)
            training_curves = load_training_curves(save_dir)
            plot_training_curves(training_curves, save_dir)