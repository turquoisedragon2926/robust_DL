import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def adversarial_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, distance='l_inf'):
    model.eval()

    # Clone the input data and require gradient computation
    x_adv = x_natural.clone().detach().requires_grad_(True)

    for _ in range(perturb_steps):
        output = model(x_adv)
        loss = F.cross_entropy(output, y)
        optimizer.zero_grad()
        loss.backward()

        # For 'l_inf' distance, the gradient is used to create adversarial examples
        if distance == 'l_inf':
            x_adv_grad = torch.sign(x_adv.grad.data)
            x_adv.data = x_adv.data + step_size * x_adv_grad

            # Clip x_adv to be within epsilon of x_natural, ensuring it stays within the valid data range
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon).clamp(0, 1.0)

        # Zero out gradients for the next iteration
        model.zero_grad()
        x_adv.grad.data.zero_()

    model.train()
    x_adv = Variable(x_adv.data, requires_grad=False)
    adv_output = model(x_adv)
    loss_adv = F.cross_entropy(adv_output, y)

    return loss_adv
