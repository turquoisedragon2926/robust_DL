import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Command line arguments for experiment script.')

    # Common Parameters
    parser.add_argument('--mode_type', type=str, default="train", choices=['eval', 'train'], help='Mode type: train or eval only')
    parser.add_argument('--attack_type', type=str, default="identity", choices=['identity', 'pgd'], help='Attack type: identity or pgd only')
    parser.add_argument('--model_type', type=str, default="alexnet", choices=['alexnet', 'resnet18'], help='Model type: alexnet')
    parser.add_argument('--train_dataset', type=str, default="cifar10", choices=['cifar10', 'imagenet'], help='Training Dataset type (default: cifar10)')
    parser.add_argument('--eval_dataset', type=str, default="cifar10C", choices=['cifar10C', 'imagenetC'], help='Evaluation Dataset type (default: cifar10C)')
    parser.add_argument('--loss_type', type=str, default="adaptive", choices=['trades', 'adaptive', 'ce'], help='Loss type: trades or custom or ce')
    parser.add_argument('--train_noise', type=str, default='gaussian', choices=['gaussian', 'uniform', 'shot', 'blur', 'random'], help='Type of noise to use while training (default: gaussian)')
    parser.add_argument('--eval_noise', type=str, default='gaussian_noise.npy', help='Type of noise (default: gaussian_noise.npy)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--valid_size', type=float, default=0.2, help='Validation dataset ratio')
    parser.add_argument('--eval_interval', type=int, default=1, help='Evaluation interval for robustness plot')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='Path to model checkpoint (optional)')
    parser.add_argument('--optimizer_checkpoint', type=str, default=None, help='Path to optimizer checkpoint (optional)')

    # Trades Loss Parameters
    parser.add_argument('--alpha', type=float, default=2.0, help='Alpha value (default: 2)')

    # Adaptive Loss Parameters
    parser.add_argument('--severity', type=float, default=0.05, help='Severity value (default: 0.05)')
    parser.add_argument('--w_noise', type=float, default=0.1, help='Weightage of our noise value (default: 0.1)')
    parser.add_argument('--tau1', type=int, default=10, help='Tau1 for norm clipping (default: 10)')
    parser.add_argument('--tau2', type=int, default=-10, help='Tau2 for norm clipping (default: -10)')
    
    return parser.parse_args()

def get_config_id(args, default=['mode_type', 'model_checkpoint', 'optimizer_checkpoint'], disclude=[]):
     return "_".join([f"{arg}={getattr(args, arg)}" for arg in vars(args) if arg not in default + disclude and getattr(args, arg) is not None])

def load_from_key(path, key):
    if os.path.exists(path):
        with open(path, 'r') as fp:
            data = json.load(fp)
        return data.get(key)
    return None

def save_to_key(path, key, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {}
    if os.path.exists(path):
        with open(path, 'r') as fp:
            data = json.load(fp)
    
    data[key] = value

    with open(path, 'w') as fp:
        json.dump(data, fp)
