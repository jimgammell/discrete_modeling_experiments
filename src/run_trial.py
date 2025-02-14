import os
from copy import copy
import yaml
import argparse

from common import *
from trials import WeightQuantizationTrial

AVAILABLE_DATASETS = ['mnist', 'cifar10']
TRIAL_TYPES = ['weight-quantization']

def run_weight_quantization_trial(args, config):
    dataset_name = args.dataset
    seed_count = args.seed_count
    save_dir = args.save_dir
    trial = WeightQuantizationTrial(dataset_name, save_dir, seed_count, **config['weight_quantization'])
    trial()

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='trial_type', required=True)
    weight_quantization_parser = subparsers.add_parser('weight-quantization')
    weight_quantization_parser.add_argument('--dataset', action='store', choices=AVAILABLE_DATASETS)
    weight_quantization_parser.add_argument('--config-name', action='store', default=None)
    weight_quantization_parser.add_argument('--seed-count', action='store', type=int, default=1)
    weight_quantization_parser.add_argument('--save-dir', type=str, default=None, action='store')
    weight_quantization_parser.set_defaults(run_trial=run_weight_quantization_trial)
    args = parser.parse_args()
    
    config_name = args.config_name if args.config_name is not None else args.dataset
    config_path = os.path.join(CONFIG_DIR, f'{config_name}.yaml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    args.run_trial(args, config)

if __name__ == '__main__':
    main()