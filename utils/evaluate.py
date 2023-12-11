import torch
from torch.autograd import Variable
from .logger import Logger

def accuracy(configuration, device):
    logger = Logger.get_instance()
    logger.log("EVALUATION")
    configuration.model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(configuration.data.test_loader):
            data, target = data.to(device), target.to(device)
            outputs = configuration.model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return 100. * correct / total

def robust_accuracy(configuration, device):
    logger = Logger.get_instance()
    logger.log("ROBUSTNESS EVALUATION")
    configuration.model.eval()
    correct = 0
    total = 0

    for data, target in configuration.data.attack_loader:
        data, target = data.to(device), target.to(device)

        X, y = Variable(data, requires_grad=True), Variable(target)
        correct_count = configuration.attack(configuration.model, X, y)
        correct += correct_count
        total += target.size(0)
    return 100. * correct / total
