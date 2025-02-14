import os
import shutil
from collections import defaultdict
from copy import copy
from typing import *
from torch.utils.data import Dataset
from lightning import Trainer as LightningTrainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from ..utils import *
from .module import Module
from .plot_things import plot_training_curves, plot_hparam_sweep
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
    
    def hparam_tune(self,
        save_dir: Union[str, os.PathLike],
        trial_count: int = 100,
        max_epochs: int = 100,
        override_kwargs: dict = {}
    ):
        if os.path.exists(os.path.join(save_dir, 'results.npz')):
            results = np.load(os.path.join(save_dir, 'results.npz'), allow_pickle=True)['arr_0'].item()
        else:
            lr_vals = [x*1e-3 for x in range(1, 11)]
            beta_1_vals = [0.5, 0.9, 0.99]
            beta_2_vals = [0.99, 0.999, 0.9999]
            eps_vals = [1e-10, 1e-8, 1e-6]
            weight_decay_vals = [0.0, 1e-4, 1e-2]
            results = defaultdict(list)
            for trial_idx in range(trial_count):
                experiment_dir = os.path.join(save_dir, f'trial_{trial_idx}')
                os.makedirs(experiment_dir, exist_ok=True)
                hparams = {
                    'lr': np.random.choice(lr_vals),
                    'beta_1': np.random.choice(beta_1_vals),
                    'beta_2': np.random.choice(beta_2_vals),
                    'eps': np.random.choice(eps_vals),
                    'weight_decay': np.random.choice(weight_decay_vals)
                }
                override_kwargs.update(hparams)
                try:
                    self.run(save_dir=experiment_dir, max_epochs=max_epochs, override_module_kwargs=override_kwargs)
                except:
                    print(f'Failed trial with hparams {hparams}')
                np.savez(os.path.join(experiment_dir, 'hparams.npz'), hparams)
                training_curves = load_training_curves(experiment_dir)
                for key, val in hparams.items():
                    results[key].append(val)
                optimal_idx = np.argmax(training_curves['val_acc'])
                results['es_acc'].append(training_curves['val_acc'][-1][optimal_idx])
                results['final_acc'].append(training_curves['val_acc'][-1][-1])
                results['es_loss'].append(training_curves['val_loss'][-1][optimal_idx])
                results['final_loss'].append(training_curves['val_loss'][-1][-1])
                results['es_train_loss'].append(training_curves['train_loss'][-1][optimal_idx])
                results['final_train_loss'].append(training_curves['train_loss'][-1][-1])
            np.savez(os.path.join(save_dir, 'results.npz'), results)
        return plot_hparam_sweep(results, save_dir)