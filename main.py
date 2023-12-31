from __future__ import print_function
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from losses.trades import trades_loss
from losses.ce import ce_loss
from losses.adaptive import adaptive_loss

from models.wideresnet import *
from models.resnet import *
from models.small_cnn import *
from models.AlexNet import AlexNet

from attacks.identity import identity_attack
from utils.logger import Logger

from utils.components import Configuration, Data
from utils.data_loader import DataLoaderFactory
from utils.train import train
from utils.evaluate import accuracy, robust_accuracy
from utils.utils import save_to_key, parse_args, get_config_id

def general_trades_loss_fn(beta=6.0, epsilon=0.3, step_size=0.007, num_steps=10):
  def trades_loss_fn(model, data, target, optimizer):
    return trades_loss(model=model, x_natural=data, y=target, optimizer=optimizer, step_size=step_size,
                      epsilon=epsilon, perturb_steps=num_steps, beta=beta, distance='l_inf')
  return trades_loss_fn

def general_adaptive_loss_fn(noise_model, train_noise, severity, w_noise, tau1, tau2):
  def adaptive_loss_fn(model, data, target, optimizer):
    return adaptive_loss(model, data, target, train_noise, noise_model, severity=severity, w_noise=w_noise, tau1=tau1, tau2=tau2)

  return adaptive_loss_fn

def main():
    args = parse_args()
    config_id = get_config_id(args)

    Logger.initialize(log_filename=f"{config_id}.txt")
    logger = Logger.get_instance()

    data_loader = DataLoaderFactory(root='data', valid_size=args.valid_size)
    train_loader, valid_loader, test_loader = data_loader.get_cifar10_loaders()

    cifar10c_attack_loader = data_loader.get_cifar10c_attack_loader(args.eval_noise)

    # Pass these loaders to the Data class
    cifar10_c_data = Data(train_loader, valid_loader, test_loader, cifar10c_attack_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_type == 'alexnet':
        model = AlexNet().to(device)
    else:
       logger.log("MODEL TYPE NOT SUPPORTED")
       sys.exit(1)

    # Define our learnable noise model and the optimizer
    noise_model = nn.Sequential(nn.Linear(1,10, bias=True),nn.Sigmoid()).to(device)
    optimizer = optim.SGD(list(model.parameters()) + list(noise_model.parameters()), lr=0.01, momentum=0.9)

    # Load checkpoints if needed
    if args.model_checkpoint is not None:
        model_pt = os.path.join("results", "models", args.model_checkpoint)
        model.load_state_dict(torch.load(model_pt))

    if args.optimizer_checkpoint is not None:
        optimizer_tar = os.path.join("results", "optimizers", args.optimizer_checkpoint)
        optimizer.load_state_dict(torch.load(optimizer_tar))
    
    # Define loss function based on args.loss_type
    if args.loss_type == 'trades':
        beta = 1 / args.alpha
        loss_fn = general_trades_loss_fn(beta=beta)
    elif args.loss_type == 'ce':
        loss_fn = ce_loss
    elif args.loss_type == 'adaptive':
        loss_fn = general_adaptive_loss_fn(noise_model, args.train_noise, args.severity, args.w_noise, args.tau1, args.tau2)
    else:
        logger.log("Loss Type not supported")
        sys.exit(1)

    # Create Configuration instance
    configuration = Configuration(cifar10_c_data, model, optimizer, loss_fn, identity_attack, config_id)

    if args.mode_type == "train":
        final_loss, total_time = train(configuration, args, device)
        save_to_key('results/metrics/final_loss.json', configuration.id, final_loss)
        save_to_key('results/metrics/train_time.json', configuration.id, total_time)

    natural_accuracy = accuracy(configuration, device)
    robustness_accuracy = robust_accuracy(configuration, device)

    save_to_key('results/metrics/natural_accuracy.json', configuration.id, natural_accuracy)
    save_to_key('results/metrics/robustness_accuracy.json', configuration.id, robustness_accuracy)

if __name__ == "__main__":
    main()
