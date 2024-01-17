import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

def pgd_attack(model, X, y, epsilon=0.031, num_steps=20, step_size=0.003):
    """
    Perform the PGD attack on a batch of images and return the model's accuracy on these adversarial examples.
    
    :param model: The model being attacked.
    :param X: Batch of input images.
    :param y: Corresponding true labels for the input images.
    :param epsilon: The maximum perturbation allowed.
    :param num_steps: Number of steps in the PGD attack.
    :param step_size: Step size for each iteration of the attack.
    :return: Accuracy of the model on the adversarial examples.
    """
    X_pgd = Variable(X.data, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        
        # Update adversarial examples
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        # Project back to the epsilon-ball and clip to valid image range
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    # Evaluate on adversarial examples
    out = model(X_pgd)
    acc = (out.data.max(1)[1] == y.data).float().mean()  # mean() for average accuracy over the batch
    return acc.item()
