import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def adversarial_loss(model, x_natural, y, alpha, k, epsilon=0.031, optimizer=None):

    def perturb(x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

    criterion = nn.CrossEntropyLoss()
    adv = perturb(x_natural, y)
    adv_outputs = model(adv)
    loss = criterion(adv_outputs, y)
    
    return loss
