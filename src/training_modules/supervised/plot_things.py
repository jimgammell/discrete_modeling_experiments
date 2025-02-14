import os
import numpy as np
from matplotlib import pyplot as plt

from common import *

def plot_hparam_sweep(results, save_dir):
    result_names = ['es_acc', 'final_acc', 'es_loss', 'final_loss', 'es_train_loss', 'final_train_loss']
    hparam_names = ['lr', 'beta_1', 'beta_2', 'eps', 'weight_decay']
    best_es_acc = -np.inf
    for idx in range(len(results['es_acc'])):
        es_acc = results['es_acc'][idx]
        if es_acc > best_es_acc:
            best_es_acc = es_acc
            selected_settings = {key: results[key][idx] for key in hparam_names}
            selected_results = {key: results[key][idx] for key in result_names}
    fig, axes = plt.subplots(len(hparam_names), len(result_names), figsize=(PLOT_WIDTH*len(result_names), PLOT_WIDTH*len(hparam_names)))
    for row_idx, (hparam_name, axes_row) in enumerate(zip(hparam_names, axes)):
        for col_idx, (result_name, ax) in enumerate(zip(result_names, axes_row)):
            hparam_vals = results[hparam_name]
            distinct_hparam_vals = list(set(hparam_vals))
            if all(isinstance(x, int) or isinstance(x, float) for x in hparam_vals):
                distinct_hparam_vals.sort()
            result_vals = results[result_name]
            label_to_num = {hparam_name: idx for idx, hparam_name in enumerate(distinct_hparam_vals)}
            xx = [label_to_num[x] for x in hparam_vals]
            ax.plot(xx, result_vals, color='blue', marker='.', linestyle='none', markersize=5, **PLOT_KWARGS)
            ax.plot([label_to_num[selected_settings[hparam_name]]], [selected_results[result_name]], color='red', marker='.', linestyle='none', markersize=5, **PLOT_KWARGS)
            ax.set_xticks(list(label_to_num.values()))
            if hparam_name in ['lr', 'eps', 'weight_decay']:
                ticklabels = [f'{x:.1e}' for x in label_to_num.keys()]
            else:
                ticklabels = [str(x) for x in label_to_num.keys()]
            ax.set_xticklabels(ticklabels, rotation=45, ha='right')
            ax.set_xlabel(hparam_name.replace('_', r'\_'))
            ax.set_ylabel(result_name.replace('_', r'\_'))
            if 'loss' in result_name:
                ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'hparam_sweep.png'), **SAVEFIG_KWARGS)
    plt.close(fig)
    return selected_settings

def plot_training_curves(training_curves, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(3*PLOT_WIDTH, PLOT_WIDTH))
    axes[0].plot(*training_curves['train_loss'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_loss'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    axes[1].plot(training_curves['train_acc'][0], 1-training_curves['train_acc'][1], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
    axes[1].plot(training_curves['val_acc'][0], 1-training_curves['val_acc'][1], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    axes[2].plot(*training_curves['train_rms_grad'], color='blue', linestyle='-', **PLOT_KWARGS)
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Loss')
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Error')
    axes[2].set_xlabel('Training step')
    axes[2].set_ylabel('RMS gradient')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'training_curves.png'), **SAVEFIG_KWARGS)