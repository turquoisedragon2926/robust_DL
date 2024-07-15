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

    severity_accuracies = {}

    natural_accuracies = []
    robustness_accuracies = []

    config_id = get_config_id(args)
    model_pt = os.path.join("results", "models", f"{config_id}.pt")
    model.load_state_dict(torch.load(model_pt))

    if args.attack_type == 'identity':
        attack = identity_attack
    elif args.attack_type == 'pgd':
        attack = pgd_attack
    
    data = Data(train_loader, valid_loader, test_loader, None)
    configuration = Configuration(data, model, None, None, attack, config_id)

    # Initialize a list to keep track of all robustness_accuracies for this severity across eval_noises
    severity_robustness_accuracies = []

    for eval_noise in eval_noises:
        logger.log(f"ON EVALUATION NOISE = {eval_noise}")

        if eval_noise == 'none':
            natural_accuracy = load_from_key(natural_accuracy_path, configuration.id)
            if natural_accuracy is None:
                natural_accuracy = accuracy(configuration, device)
                save_to_key(natural_accuracy_path, configuration.id, natural_accuracy)
            natural_accuracies.append(natural_accuracy)
            severity_robustness_accuracies.append(natural_accuracy)
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

        # Force eval each time
        robustness_accuracy = None # load_from_key(robustness_accuracy_path, configuration.id)
        if robustness_accuracy is None:
            robustness_accuracy = robust_accuracy(configuration, device)
            save_to_key(robustness_accuracy_path, configuration.id, robustness_accuracy)
        severity_robustness_accuracies.append(robustness_accuracy)
        configuration.attack = attack
        args.eval_noise = default_noise

    # Disclude natural and adversarial accuracy from this 
    robustness_accuracies.append(sum(severity_robustness_accuracies[2:]) / len(severity_robustness_accuracies[1:]))
    severity_accuracies["metric"] = severity_robustness_accuracies

    total_robustness_accuracies["train_noise"] = robustness_accuracies
    total_natural_accuracies["train_noise"] = natural_accuracies

    configuration.id = get_config_id(args, disclude=['eval_noise'])

    # plotter.plot_severity_vs_robustness(severities, natural_accuracies, robustness_accuracies, train_noise, plot_name=f"{configuration.id}_severity_vs_robustness.png")
    # plotter.plot_eval_noise_bar_chart(eval_noises, severity_accuracies, train_noise, plot_name=f"{configuration.id}_noise_vs_robustness.png")
    
    # plotter.plot_combined_severity_vs_robustness(severities, total_robustness_accuracies, train_noises, plot_name=f"{configuration.id}_combined_severity_vs_robustness.png")
    # plotter.plot_combined_severity_vs_robustness(severities, total_natural_accuracies, train_noises, plot_name=f"{configuration.id}_combined_severity_vs_natural.png", robust=False)

    # plotter.plot_tradeoff(severities, total_natural_accuracies, total_robustness_accuracies, plot_name=f"{configuration.id}_tradeoff.png")
    print(natural_accuracies, severity_accuracies, total_robustness_accuracies)

if __name__ == "__main__":
    main()
