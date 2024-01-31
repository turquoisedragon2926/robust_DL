from __future__ import print_function
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.wideresnet import *
from models.resnet import *
from models.small_cnn import *
from models.AlexNet import AlexNet

from attacks.identity import identity_attack
from attacks.pgd import pgd_attack
from utils.logger import Logger
from utils.plotter import Plotter

from utils.components import Configuration, Data
from utils.data_loader import DataLoaderFactory
from utils.evaluate import accuracy, robust_accuracy
from utils.utils import parse_args, get_config_id, save_to_key, load_from_key

def main():
    args = parse_args()

    Logger.initialize(log_filename=f"plotting.txt")
    logger = Logger.get_instance()
    plotter = Plotter('results/plots/eval')

    data_loader = DataLoaderFactory(root='data', valid_size=args.valid_size, train_dataset=args.train_dataset, eval_dataset=args.eval_dataset)
    train_loader, valid_loader, test_loader = data_loader.get_data_loaders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_noises = ["none", "adversarial", "saturate.npy", "spatter.npy", "gaussian_blur.npy", "speckle_noise.npy", "jpeg_compression.npy", "pixelate.npy", "elastic_transform.npy", "contrast.npy", "brightness.npy", "fog.npy", "frost.npy", "snow.npy", "zoom_blur.npy", "motion_blur.npy", "defocus_blur.npy", "impulse_noise.npy", "shot_noise.npy", "gaussian_noise.npy"]

    w_noises = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    alphas = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0] 
    # alphas = [3.0, 2.0, 1.0, 0.5, 0.25, 0.1] # If We want betas to be increasing

    if args.model_type == 'alexnet':
        model = AlexNet().to(device)
    elif args.model_type == 'resnet18':
        model = ResNet18().to(device)
    else:
       logger.log("MODEL TYPE NOT SUPPORTED")
       sys.exit(1)

    natural_accuracy_path = 'results/metrics/natural_accuracy.json'
    robustness_accuracy_path = 'results/metrics/robustness_accuracy.json'

    total_robustness_accuracies = {}
    total_natural_accuracies = {}

    w_noise_accuracies = {}
    alpha_accuracies = {}

    trades_adversarial_accuracies = []
    adaptive_adversarial_accuracies = []

    natural_accuracies = []
    robustness_accuracies = []

    args.train_noise = "uniform"
    args.loss_type = "adaptive"
    args.lr = 0.03 # From emperical testing

    for w_noise in w_noises:
        logger.log(f"ON W_NOISE = {w_noise}")

        args.w_noise = w_noise
        config_id = get_config_id(args)
        model_pt = os.path.join("results", "models", f"{config_id}.pt")
        model.load_state_dict(torch.load(model_pt))
        if args.attack_type == 'identity':
            attack = identity_attack
        elif args.attack_type == 'pgd':
            attack = pgd_attack

        data = Data(train_loader, valid_loader, test_loader, None)
        configuration = Configuration(data, model, None, None, attack, config_id)
        # Initialize a list to keep track of all robustness_accuracies for this w_noise across eval_noises
        w_noise_robustness_accuracies = []
        for eval_noise in eval_noises:
            logger.log(f"ON EVALUATION NOISE = {eval_noise}")
            if eval_noise == 'none':
                natural_accuracy = load_from_key(natural_accuracy_path, configuration.id)
                if natural_accuracy is None:
                    natural_accuracy = accuracy(configuration, device)
                    save_to_key(natural_accuracy_path, configuration.id, natural_accuracy)
                natural_accuracies.append(natural_accuracy)
                w_noise_robustness_accuracies.append(natural_accuracy)
                continue
            attack_loader = data_loader.get_attack_loader(eval_noise)
            # TODO: Abstract attack and eval noise to common framework
            if eval_noise == 'adversarial':
                configuration.attack = pgd_attack
            # Change noise to save properly in json
            default_noise = args.eval_noise
            args.eval_noise = eval_noise
            configuration.id = get_config_id(args)
            # Load up the new evaluation data
            data.attack_loader = attack_loader
            configuration.data = data
            robustness_accuracy = load_from_key(robustness_accuracy_path, configuration.id)
            if robustness_accuracy is None:
                robustness_accuracy = robust_accuracy(configuration, device)
                save_to_key(robustness_accuracy_path, configuration.id, robustness_accuracy)
            w_noise_robustness_accuracies.append(robustness_accuracy)
            configuration.attack = attack
            args.eval_noise = default_noise
        # Disclude natural and adversarial accuracy from this
        robustness_accuracies.append(sum(w_noise_robustness_accuracies[2:]) / len(w_noise_robustness_accuracies[1:]))
        w_noise_accuracies[w_noise] = w_noise_robustness_accuracies
        adaptive_adversarial_accuracies.append(w_noise_robustness_accuracies[1])

    total_robustness_accuracies["adaptive"] = robustness_accuracies
    total_natural_accuracies["adaptive"] = natural_accuracies

    # Robustnes acc reset
    natural_accuracies = []
    robustness_accuracies = []


    args.w_noise = 0.1 # resetting
    args.train_noise = "gaussian" # resetting
    args.loss_type = "trades"
    args.lr = 0.01 # From emperical testing

    for alpha in alphas:
        logger.log(f"ON alpha = {alpha}")

        args.alpha = alpha
        config_id = get_config_id(args)
        model_pt = os.path.join("results", "models", f"{config_id}.pt")
        model.load_state_dict(torch.load(model_pt))
        if args.attack_type == 'identity':
            attack = identity_attack
        elif args.attack_type == 'pgd':
            attack = pgd_attack

        data = Data(train_loader, valid_loader, test_loader, None)
        configuration = Configuration(data, model, None, None, attack, config_id)
        # Initialize a list to keep track of all robustness_accuracies for this alpha across eval_noises
        alpha_robustness_accuracies = []
        for eval_noise in eval_noises:
            logger.log(f"ON EVALUATION NOISE = {eval_noise}")
            if eval_noise == 'none':
                natural_accuracy = load_from_key(natural_accuracy_path, configuration.id)
                if natural_accuracy is None:
                    natural_accuracy = accuracy(configuration, device)
                    save_to_key(natural_accuracy_path, configuration.id, natural_accuracy)
                natural_accuracies.append(natural_accuracy)
                alpha_robustness_accuracies.append(natural_accuracy)
                continue
            attack_loader = data_loader.get_attack_loader(eval_noise)
            # TODO: Abstract attack and eval noise to common framework
            if eval_noise == 'adversarial':
                configuration.attack = pgd_attack
            # Change noise to save properly in json
            default_noise = args.eval_noise
            args.eval_noise = eval_noise
            configuration.id = get_config_id(args)
            # Load up the new evaluation data
            data.attack_loader = attack_loader
            configuration.data = data
            robustness_accuracy = load_from_key(robustness_accuracy_path, configuration.id)
            if robustness_accuracy is None:
                robustness_accuracy = robust_accuracy(configuration, device)
                save_to_key(robustness_accuracy_path, configuration.id, robustness_accuracy)
            alpha_robustness_accuracies.append(robustness_accuracy)
            configuration.attack = attack
            args.eval_noise = default_noise
        # Disclude natural and adversarial accuracy from this
        robustness_accuracies.append(sum(alpha_robustness_accuracies[2:]) / len(alpha_robustness_accuracies[1:]))
        alpha_accuracies[alpha] = alpha_robustness_accuracies
        trades_adversarial_accuracies.append(alpha_robustness_accuracies[1])

    total_robustness_accuracies["trades"] = robustness_accuracies
    total_natural_accuracies["trades"] = natural_accuracies

    configuration.id = get_config_id(args, disclude=['eval_noise'])

    # # Prepare the data for plotting to use with plot_accuracy_vs_parameters in Plotter
    # alpha_accuracies = {
    #     'Adversarial Accuracy (TRADES)': trades_adversarial_accuracies,  # Assuming this is a list of accuracies for each alpha
    #     'Natural Accuracy (TRADES)': total_natural_accuracies['trades'],  # List of natural accuracies for each alpha
    #     'Average Robustness Accuracy (TRADES)': total_robustness_accuracies['trades']  # List of robustness accuracies for each alpha
    # }
    # w_noise_accuracies = {
    #     'Adversarial Accuracy (Adaptive)': adaptive_adversarial_accuracies,  # Assuming this is a list of accuracies for each w_noise
    #     'Natural Accuracy (Adaptive)': total_natural_accuracies['adaptive'],  # List of natural accuracies for each w_noise
    #     'Average Robustness Accuracy (Adaptive)': total_robustness_accuracies['adaptive']  # List of robustness accuracies for each w_noise
    # }

    adversarial_accuracies_to_plot = {
        "TRADES": {
            "x": total_natural_accuracies['trades'],
            "y": trades_adversarial_accuracies,
            "keys": alphas
        },
        "ADAPTIVE": {
            "x": total_natural_accuracies['adaptive'],
            "y": adaptive_adversarial_accuracies,
            "keys": w_noise
        }
    }

    robustness_accuracies_to_plot = {
        "TRADES": {
            "x": total_natural_accuracies['trades'],
            "y": total_robustness_accuracies['trades'],
            "keys": alphas
        },
        "ADAPTIVE": {
            "x": total_natural_accuracies['adaptive'],
            "y": total_robustness_accuracies['adaptive'],
            "keys": w_noise
        }
    }

    # Call the new plotting function
    plotter.plot_accuracy_vs_accuracy(adversarial_accuracies_to_plot, plot_name=f"{configuration.id}_adversarial_vs_accuracy.png", y_axis="Adversarial Accuracy")
    plotter.plot_accuracy_vs_accuracy(robustness_accuracies_to_plot, plot_name=f"{configuration.id}_robustness_vs_accuracy.png", y_axis="Average Robustness Accuracy")

if __name__ == "__main__":
    main()
