from torch.utils.data import Dataset

class Data:
  def __init__(self, train_loader, valid_loader, test_loader, attack_loader):
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.test_loader = test_loader
    self.attack_loader = attack_loader

class Configuration:
  def __init__(self, data, model, optimizer, loss_fn, attack, id=None):
    self.data = data
    self.model = model
    self.loss_fn = loss_fn
    self.attack = attack
    self.optimizer = optimizer

    self.id = id

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
