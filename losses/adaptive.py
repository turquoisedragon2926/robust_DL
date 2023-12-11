import torch
from torch import linalg
import torch.nn.functional as F

def log_prob_drichlet(x,theta):
    D = torch.distributions.dirichlet.Dirichlet(theta)
    return D.log_prob(x)

def gaussian_noise_layer(data, severity=0.05):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    noise = torch.randn(data.size()) * severity
    noise = torch.Tensor(noise).to(device)

    noisy_data = data+noise
    noise_norm = linalg.vector_norm(noise, dim = (1,2,3)).reshape((-1,1)) # improve this line of code later
    return noisy_data, noise_norm

def norm_clamp(logit, tau1, tau2):
    logit_norm = linalg.vector_norm(logit, dim=1, keepdim=True)
    scale = torch.clamp(logit_norm, min=tau2, max=tau1) / logit_norm
    return logit * scale

def adaptive_loss(model, x_natural, y, noise_model, severity, w_noise, tau1, tau2):
    logits = model(x_natural)
    logits = norm_clamp(logits, tau1, tau2)

    p = torch.nn.functional.softmax(logits, dim=1)

    _, noise_norm = gaussian_noise_layer(x_natural, severity)
    noise_weight = noise_model(noise_norm)
    y_one_hot = torch.zeros_like(logits)
    y_one_hot[:,y]=1

    # Different variance for each class
    noise_weight_for_class = noise_weight[torch.arange(logits.size(0)), y]
    
    loss_noisy = log_prob_drichlet(p, noise_weight_for_class.unsqueeze(1) * y_one_hot + 1).mean()
    loss_normal = F.cross_entropy(logits , y)

    return (1-w_noise)*loss_normal+w_noise*loss_noisy
