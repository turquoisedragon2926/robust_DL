import torch.nn.functional as F

def ce_loss(model, data, target, optimizer):
    return F.cross_entropy(model(data), target)
