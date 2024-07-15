import torch
from torch import linalg
import torch.nn.functional as F
from .noises import NoiseFunctionFactory

def log_prob_drichlet(x,theta):
    D = torch.distributions.dirichlet.Dirichlet(theta)
    return D.log_prob(x)

def norm_clamp(logit, tau1, tau2):
    logit_norm = linalg.vector_norm(logit, dim=1, keepdim=True)
    scale = torch.clamp(logit_norm, min=tau2, max=tau1) / logit_norm
    return logit * scale

def adaptive_loss(model, x_natural, y, train_noise, noise_model, severity, w_noise, tau1, tau2, num_samples=10):
    logits = model(x_natural)
    logits = norm_clamp(logits, tau1, tau2)
    
    p = F.softmax(logits, dim=1)
    y_one_hot = torch.zeros_like(logits)
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    
    max_loss_noisy = None
    for _ in range(num_samples):
        noise_fn = NoiseFunctionFactory.get_noise_function(train_noise, severity)
        _, noise_norm = noise_fn.add_noise(x_natural)
        noise_weight = noise_model(noise_norm)
        noise_weight_for_class = noise_weight[torch.arange(logits.size(0)), y]
        current_loss_noisy = log_prob_drichlet(p, noise_weight_for_class.unsqueeze(1) * y_one_hot + 1)
        
        if max_loss_noisy is None:
            max_loss_noisy = current_loss_noisy
        else:
            max_loss_noisy = torch.maximum(max_loss_noisy, current_loss_noisy)
    
    loss_noisy = max_loss_noisy.mean()
    loss_normal = F.cross_entropy(logits, y)
    
    return (1 - w_noise) * loss_normal + w_noise * loss_noisy
