from __future__ import print_function
import os
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from losses.trades import trades_loss
import copy
import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
from models.small_cnn import *
from torch.utils.data import Dataset, DataLoader
from models.AlexNet import AlexNet
from torch.utils.data.sampler import SubsetRandomSampler
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch import linalg
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_prob_drichlet(x,theta):
    D = torch.distributions.dirichlet.Dirichlet(theta)
    return D.log_prob(x)

def gaussian_noise_layer(data, severity=0.05):
    # c = [0.04, 0.06, .08, .09, .10][severity - 1]
    noise = torch.randn(data.size())* severity#.to(device)

    noise =torch.Tensor(noise).to(device)
    # noise = noise*data.mean()
    # noisy_data = torch.clip(data + noise)
    noisy_data = data+noise
    # noise = noisy_data - data
    noise_norm = linalg.vector_norm(noise, dim = (1,2,3)).reshape((-1,1))  # improve this line of code later
    return noisy_data, noise_norm

def adaptive_loss(model,x_natural,y,noise_model, severity=0.05):
    logits = model(x_natural)
    p = torch.nn.functional.softmax(logits, dim=1)

    noisy_data, noise_norm = gaussian_noise_layer(x_natural)
    noise_weight = noise_model(noise_norm)#torch.div(noise_model(noise_norm), noise_model(torch.zeros_like(noise_norm)))
    y_one_hot = torch.zeros_like(logits)
    y_one_hot[:,y]=1

    # loss_noisy = log_prob_dirichlet(logits, noise_weight)
    loss_noisy = log_prob_drichlet(p,noise_weight*y_one_hot+1).mean()

    # loss_normal = F.cross_entropy(logits , y)
    return loss_noisy

def adaptive_loss_v2(model,x_natural,y,noise_model, severity=0.05, w_noise=0.2):
    logits = model(x_natural)
    p = torch.nn.functional.softmax(logits, dim=1)

    noisy_data, noise_norm = gaussian_noise_layer(x_natural)
    noise_weight = noise_model(noise_norm)#torch.div(noise_model(noise_norm), noise_model(torch.zeros_like(noise_norm)))
    y_one_hot = torch.zeros_like(logits)
    y_one_hot[:,y]=1

    # loss_noisy = log_prob_dirichlet(logits, noise_weight)
    loss_noisy = log_prob_drichlet(p,noise_weight*y_one_hot+1).mean()

    loss_normal = F.cross_entropy(logits , y)
    return (1-w_noise)*loss_normal+w_noise*loss_noisy

class Data:
  def __init__(self, train_loader, valid_loader, test_loader, attack_loader):
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.test_loader = test_loader
    self.attack_loader = attack_loader

class Model:
  model = None
  def __init__(self, id):
    self.id = id

class Loss:
  def __init__(self, loss_fn, id=None):
    self.loss_fn = loss_fn
    self.id = id

class Configuration:
  def __init__(self, data, model, optimizer, loss, attack, id=None):
    self.data = data
    self.model = model
    self.optimizer = optimizer
    self.loss = loss
    self.attack = attack

    self.id = id

  def getConfig(self):
    return self.data, self.model, self.optimizer, self.loss, self.attack

  def getId(self):
    return self.id

class CIFAR10CDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def general_trades_loss_fn(beta=6.0, epsilon=0.3, step_size=0.007, num_steps=10):
  def trades_loss_fn(model, data, target, optimizer):
    return trades_loss(model=model, x_natural=data, y=target, optimizer=optimizer, step_size=step_size,
                      epsilon=epsilon, perturb_steps=num_steps, beta=beta, distance='l_inf')
  return trades_loss_fn

def general_adaptive_loss_fn(noise_model, severity=0.05):
  def adaptive_loss_fn(model, data, target, optimizer):
    return adaptive_loss_v2(model, data, target, noise_model, severity=severity)

  return adaptive_loss_fn

def ce_loss_fn(model, data, target, optimizer):
    return F.cross_entropy(model(data), target)

def identity_attack(model, X, y):
  out = model(X)
  acc = (out.data.max(1)[1] == y.data).float().sum()
  return acc.item()

def accuracy(model, data_loader, device):
    print('EVAL')
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return 100. * correct / total

def robust_accuracy(model, attack, data_loader, device):
    print('ROBUST EVAL')
    model.eval()
    correct = 0
    total = 0

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)

        X, y = Variable(data, requires_grad=True), Variable(target)
        correct_count = attack(model, X, y)
        correct += correct_count
        total += target.size(0)
    return 100. * correct / total

def save_epoch_losses(loss_id, epoch_losses):
    os.makedirs('epoch_losses', exist_ok=True)
    file_path = os.path.join('epoch_losses', f'{loss_id}_EPOCHS={len(epoch_losses)}.txt')
    with open(file_path, 'w') as file:
        for loss in epoch_losses:
            file.write(f'{loss}\n')

def save_epoch_accuracies(loss_id, epoch_accuracies):
    os.makedirs('epoch_accuracies', exist_ok=True)
    file_path = os.path.join('epoch_accuracies', f'{loss_id}_EPOCHS={len(epoch_accuracies)}.txt')
    with open(file_path, 'w') as file:
        for loss in epoch_accuracies:
            file.write(f'{loss}\n')


def train(model, data, optimizer, loss, config, epochs, eval_interval, device):
  print('TRAINING')
  data_loader = data.train_loader
  valid_loader = data.valid_loader
  attack_loader = data.attack_loader

  model.to(device)
  best_eval_acc = 0.0
  patience = 5  # number of VAL Acc values observed after best value to stop training

  # Initialize lists to store per-epoch loss and validation accuracy
  epoch_losses = []
  eval_accuracies = []

  for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        l = loss.loss_fn(model, data, target, optimizer)
        l.backward()
        optimizer.step()
        total_loss += l.item()

        print(loss.id + f" @ EP={epoch} & Batch idx " + str(batch_idx) + " / " + str(len(data_loader) - 1) + " Loss: " + str(l.item()))

    epoch_losses.append(total_loss / len(data_loader))

    if epoch == 1 or epoch % eval_interval == 0 or epoch == epochs:
      eval_acc= accuracy(model, valid_loader, device)
      eval_accuracies.append(eval_acc)

      if (eval_acc > best_eval_acc):  # best so far so save checkpoint to restore later
        best_eval_acc = eval_acc
        patience_count = 0
        torch.save(model.state_dict(), os.path.join("weights", loss.id + f"_EPOCHS={epoch}.pt"))
        torch.save(optimizer.state_dict(), os.path.join("optimizers", loss.id + f"_EPOCHS={epoch}.tar"))
      else:
          patience_count += 1

    save_epoch_losses(loss.id, epoch_losses)
    save_epoch_accuracies(loss.id, eval_accuracies)

    # Plotting the loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.suptitle(loss.id)

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses, label='Training Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(eval_accuracies, label='Validation Accuracy')

    if epoch == epochs or patience_count >= patience:
      # Get the CIFAR 10 C evaluation accuracy and plot the horizontal line
      cifar10c_eval_acc = robust_accuracy(model, config.attack, attack_loader, device)
      plt.axhline(y=cifar10c_eval_acc, color='r', linestyle='-', label='CIFAR 10 C EVAL')

    plt.title('Validation Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(os.path.join('plots', loss.id + f'_EPOCHS={epoch}_training_validation_plot.png'))
    plt.close()

    if patience_count >= patience:
      print(f"Early Stopping!, epoch {epoch}")
      break

  return total_loss

def main():
    if len(sys.argv) < 6 or len(sys.argv) > 7:
        print("Usage: python3 experiment.py args: [modetype (eval/train), losstype (trades/custom), noisetype, hyperparam (alpha/severity), epochs, model_checkpoint_path]")
        sys.exit(1)

    # Access and print the command-line arguments
    print("Total number of arguments:", len(sys.argv))

    modetype = sys.argv[1]
    losstype = sys.argv[2]
    noisetype = sys.argv[3]
    hyperparam = float(sys.argv[4])
    epochs = int(sys.argv[5])
    model_checkpoint = None
    if len(sys.argv) > 6:
        model_checkpoint = sys.argv[6]

    valid_size=0.2
    eval_interval=10

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),])
    transform_test = transforms.Compose([
        transforms.ToTensor(),])
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform_attack = transforms.Compose([transforms.ToTensor(),])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    validset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    cifar10_train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, sampler=train_sampler, **kwargs)
    cifar10_valid_loader = torch.utils.data.DataLoader(trainset , batch_size=128, sampler=valid_sampler, **kwargs)
    cifar10_test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, **kwargs)


    transform_cifar10c = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    images = np.load(os.path.join('../data/CIFAR-10-C', noisetype))
    labels = np.load('../data/CIFAR-10-C/labels.npy')
    cifar10c_dataset = CIFAR10CDataset(data=images,labels=labels,transform=transform_cifar10c)
    cifar10c_attack_loader = DataLoader(cifar10c_dataset, batch_size=200, shuffle=False)

    cifar10_c_data = Data(cifar10_train_loader, cifar10_valid_loader,cifar10_test_loader, cifar10c_attack_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    alexnet = AlexNet().to(device)
    noise_model = nn.Sequential(nn.Linear(1,1, bias=True),nn.Sigmoid()).to(device)

    if model_checkpoint is not None:
        model_pt = os.path.join("weights", model_checkpoint)
        alexnet.load_state_dict(torch.load(model_pt))
        # Load optimizer too if needed

    optimizer = optim.SGD(list(alexnet.parameters()) + list(noise_model.parameters()), lr=0.01, momentum=0.9)

    if losstype == 'trades':
        beta = 1 / hyperparam
        id = f'CIFARC10:Alexnet:TRADES_LOSS:BETA={beta}'
        trades_loss_beta = Loss(general_trades_loss_fn(beta=beta), id)
        configuration = Configuration(cifar10_c_data, alexnet, optimizer, trades_loss_beta, identity_attack, id=trades_loss_beta.id + f":EVAL_NOISE={noisetype}")
    elif losstype == 'ce':
        id = f'CIFARC10:Alexnet:CE_LOSS'
        ce_loss = Loss(ce_loss_fn, id)
        configuration = Configuration(cifar10_c_data, alexnet, optimizer, ce_loss, identity_attack, id=ce_loss.id + f":EVAL_NOISE={noisetype}")
    elif losstype == 'custom':
        id = f'CIFARC10:Alexnet:CUSTOM_LOSS:SEVERITY={hyperparam}'
        adaptive_loss_severity=Loss(general_adaptive_loss_fn(noise_model, hyperparam), id)
        configuration = Configuration(cifar10_c_data, alexnet, optimizer, adaptive_loss_severity, identity_attack, id=adaptive_loss_severity.id + f":EVAL_NOISE={noisetype}")
    else:
       print("Loss Type not supported")
       sys.exit(1)

    data, model, optimizer, loss, attack = configuration.getConfig()

    if modetype == "train":   
        current_loss = train(model, data, optimizer, loss, configuration, epochs, eval_interval, device)

        with open('results/final_loss.json', 'r') as fp:
            final_loss = json.load(fp)

        final_loss[configuration.getId()] = current_loss

        with open('results/final_loss.json', 'w') as fp:
            json.dump(final_loss, fp)

    current_accuracy = accuracy(model, data.test_loader, device)
    current_robust_accuracy = robust_accuracy(model, attack, data.attack_loader, device)
    
    with open('results/natural_accuracy.json', 'r') as fp:
        natural_accuracy = json.load(fp)
    with open('results/robustness_accuracy.json', 'r') as fp:
        robustness_accuracy = json.load(fp)

    natural_accuracy[configuration.getId()] = current_accuracy
    robustness_accuracy[configuration.getId()] = current_robust_accuracy

    with open('results/natural_accuracy.json', 'w') as fp:
        json.dump(natural_accuracy, fp)
    with open('results/robustness_accuracy.json', 'w') as fp:
        json.dump(robustness_accuracy, fp)

if __name__ == "__main__":
    main()
