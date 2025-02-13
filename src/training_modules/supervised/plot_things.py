import os
from matplotlib import pyplot as plt

from common import *

def plot_training_curves(training_curves, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, PLOT_WIDTH))
    axes[0].plot(*training_curves['train_loss'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_loss'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    axes[1].plot(*training_curves['train_acc'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
    axes[1].plot(*training_curves['val_acc'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Loss')
    axes[0].set_yscale('log')
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Accuracy')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'training_curves.png'), **SAVEFIG_KWARGS)