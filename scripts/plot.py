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


    data_loader = DataLoaderFactory(root='data', valid_size=args.valid_size)
    train_loader, valid_loader, test_loader = data_loader.get_cifar10_loaders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_noises = ["gaussian", "random"] # ["gaussian", "uniform", "shot", "blur", "random"]
    eval_noises = ["none", "saturate.npy", "spatter.npy", "gaussian_blur.npy", "speckle_noise.npy", "jpeg_compression.npy", "pixelate.npy", "elastic_transform.npy", "contrast.npy", "brightness.npy", "fog.npy", "frost.npy", "snow.npy", "zoom_blur.npy", "motion_blur.npy", "defocus_blur.npy", "impulse_noise.npy", "shot_noise.npy", "gaussian_noise.npy"]
    severities = [0.05, 0.1] # [0.05, 0.1, 0.25, 0.5, 0.75, 1]

    if args.model_type == 'alexnet':
        model = AlexNet().to(device)
    else:
       logger.log("MODEL TYPE NOT SUPPORTED")
       sys.exit(1)

    natural_accuracy_path = 'results/metrics/natural_accuracy.json'
    robustness_accuracy_path = 'results/metrics/robustness_accuracy.json'

    for train_noise in train_noises:
        logger.log(f"EVAL STARTED FOR {train_noise} NOISE")
        severity_accuracies = {}

        natural_accuracies = []
        robustness_accuracies = []

        for severity in severities:
            logger.log(f"ON SEVERITY = {severity}")

            args.train_noise = train_noise
            args.severity = severity

            config_id = get_config_id(args)
            model_pt = os.path.join("results", "models", f"{config_id}.pt")
            model.load_state_dict(torch.load(model_pt))

            cifar10_c_data = Data(train_loader, valid_loader, test_loader, None)
            configuration = Configuration(cifar10_c_data, model, None, None, identity_attack, config_id)

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

                cifar10c_attack_loader = data_loader.get_cifar10c_attack_loader(eval_noise)
                args.eval_noise = eval_noise
                configuration.id = get_config_id(args)

                # Load up the new evaluation data
                cifar10_c_data.attack_loader = cifar10c_attack_loader
                configuration.data = cifar10_c_data

                robustness_accuracy = load_from_key(robustness_accuracy_path, configuration.id)
                if robustness_accuracy is None:
                    robustness_accuracy = robust_accuracy(configuration, device)
                    save_to_key(robustness_accuracy_path, configuration.id, robustness_accuracy)
                severity_robustness_accuracies.append(robustness_accuracy)

            robustness_accuracies.append(sum(severity_robustness_accuracies) / len(severity_robustness_accuracies))
            severity_accuracies[severity] = severity_robustness_accuracies

        configuration.id = get_config_id(args, disclude=['eval_noise'])
        plotter.plot_severity_vs_robustness(severities, natural_accuracies, robustness_accuracies, train_noise, plot_name=f"{configuration.id}_severity_vs_robustness.png")
        plotter.plot_eval_noise_bar_chart(eval_noises, severity_accuracies, train_noise, plot_name=f"{configuration.id}_noise_vs_robustness.png")

if __name__ == "__main__":
    main()
