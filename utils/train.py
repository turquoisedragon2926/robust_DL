import os
import time
import torch
from .evaluate import accuracy, robust_accuracy
from .logger import Logger
from .plotter import Plotter

def train(configuration, args, device):
  logger = Logger.get_instance()
  logger.log('TRAINING')

  data_loader = configuration.data.train_loader

  configuration.model.to(device)
  best_eval_acc = 0.0
  total_time = 0.0
  total_loss = 0.0
  patience = 5  # number of VAL Acc values observed after best value to stop training

  # Initialize lists to store per-epoch loss and validation accuracy
  epoch_losses = []
  eval_accuracies = []

  plotter = Plotter("results/plots/train")
  for epoch in range(1, args.epochs+1):

    configuration.model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        configuration.optimizer.zero_grad()
        l = configuration.loss_fn(configuration.model, data, target, configuration.optimizer)
        l.backward()
        configuration.optimizer.step()
        total_loss += l.item()

        logger.log(configuration.id + f" @ EP={epoch} & Batch idx " + str(batch_idx) + " / " + str(len(data_loader) - 1) + " Loss: " + str(l.item()))

    end_time = time.time()
    total_time += end_time - start_time 

    epoch_losses.append(total_loss / len(data_loader))

    if epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.epochs:
      eval_acc = accuracy(configuration, device, valid=True)
      eval_accuracies.append(eval_acc)

      if (eval_acc > best_eval_acc):  # best so far so save checkpoint to restore later
        best_eval_acc = eval_acc
        patience_count = 0

        os.makedirs(os.path.join("results", "models"), exist_ok=True)
        os.makedirs(os.path.join("results", "optimizers"), exist_ok=True)
        
        torch.save(configuration.model.state_dict(), os.path.join("results", "models", configuration.id + '.pt'))
        torch.save(configuration.optimizer.state_dict(), os.path.join("results", "optimizers", configuration.id + '.tar'))
      else:
          patience_count += 1

    cifar10c_eval_acc = None
    if epoch == args.epochs or patience_count >= patience:
        # Get the CIFAR 10 C evaluation accuracy and plot the horizontal line
        cifar10c_eval_acc = robust_accuracy(configuration, device)
    
    plotter.plot_loss_accuracy(epoch_losses, eval_accuracies, cifar10c_eval_acc, configuration.id + '.png')

    if patience_count >= patience:
      logger.log(f"Early Stopping!, epoch {epoch}")
      break

  return total_loss, total_time
