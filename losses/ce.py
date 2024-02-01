import torch.nn.functional as F
from .noises import NoiseFunctionFactory

def ce_loss(model, data, target, train_noise, severity):

    noise_fn = NoiseFunctionFactory.get_noise_function(train_noise, severity)
    noisy_data, _ = noise_fn.add_noise(data)

    return F.cross_entropy(model(noisy_data), target)
