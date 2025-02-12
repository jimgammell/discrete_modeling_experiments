import os
import torch

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    gpu_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = 10*gpu_properties.major + gpu_properties.minor
    if arch >= 70:
        torch.set_float32_matmul_precision('high')

PLOT_WIDTH = 4
PLOT_KWARGS = {'rasterized': True}
SAVEFIG_KWARGS = {'dpi': 300}

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
OUTPUT_DIR = os.path.join(PROJ_DIR, 'outputs')
CONFIG_DIR = os.path.join(PROJ_DIR, 'config')
RESOURCE_DIR = os.path.join(PROJ_DIR, 'resources')
MNIST_DIR = os.path.join(RESOURCE_DIR, 'mnist')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RESOURCE_DIR, exist_ok=True)
os.makedirs(MNIST_DIR, exist_ok=True)