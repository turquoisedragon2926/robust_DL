from torch.utils.data import Dataset
import json
from PIL import Image
from torch.utils.data import Dataset
import os
import torch
from losses.augmix import single_image_aug

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
  
class AttackDataset(Dataset):
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
class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None, n_classes=10, max_samples_per_class=None, noise="gaussian"):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        self.n_classes = n_classes
        self.max_samples_per_class = max_samples_per_class
        
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)

        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        class_sample_counts = {i: 0 for i in range(n_classes)}  
        
        if split == "eval":
            samples_dir = os.path.join(root, noise, "1")
        
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                if target < self.n_classes:
                    syn_folder = os.path.join(samples_dir, syn_id)
                    for sample in os.listdir(syn_folder):
                        if self.max_samples_per_class and class_sample_counts[target] >= self.max_samples_per_class:
                            continue
                        sample_path = os.path.join(syn_folder, sample)
                        self.samples.append(sample_path)
                        self.targets.append(target)
                        class_sample_counts[target] += 1
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                if target < self.n_classes:
                    if self.max_samples_per_class and class_sample_counts[target] >= self.max_samples_per_class:
                        continue
                    sample_path = os.path.join(samples_dir, entry)
                    self.samples.append(sample_path)
                    self.targets.append(target)
                    class_sample_counts[target] += 1
            elif split == "eval":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                if target < self.n_classes:
                    syn_folder = os.path.join(samples_dir, syn_id)
                    for sample in os.listdir(syn_folder):
                        if self.max_samples_per_class and class_sample_counts[target] >= self.max_samples_per_class:
                            continue
                        sample_path = os.path.join(syn_folder, sample)
                        self.samples.append(sample_path)
                        self.targets.append(target)
                        class_sample_counts[target] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]
    
class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return single_image_aug(x, self.preprocess), y
    else:
      im_tuple = (
            self.preprocess(x), 
            single_image_aug(x, self.preprocess),
            single_image_aug(x, self.preprocess)
      )
    print(f"Loading Data Point {i}")
    return im_tuple, y

  def __len__(self):
    return len(self.dataset)
