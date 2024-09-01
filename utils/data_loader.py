import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
# from .components import AttackDataset, ImageNetKaggle

from torch.utils.data import Dataset
import json
from PIL import Image
from torch.utils.data import Dataset
import os

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
    def __init__(self, root, split, transform=None, n_classes=10, max_samples_per_class=3):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


class DataLoaderFactory:
    def __init__(self, root, valid_size, train_dataset, eval_dataset, batch_size=128, num_workers=1, pin_memory=True):
        self.root = root
        self.valid_size = valid_size
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_data_loaders(self):
        if self.train_dataset == 'cifar10':
            trainset, validset, testset = self.get_cifar10_loaders()
        elif self.train_dataset == 'imagenet':
            trainset, validset, testset = self.get_imagenet_loaders()
        else:
            raise ValueError("Dataset not implemented")

        # Create samplers for validation split
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # Create data loaders
        train_loader = DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)
        valid_loader = DataLoader(validset, batch_size=self.batch_size, sampler=valid_sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)
        test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

        return train_loader, valid_loader, test_loader

    def get_attack_loader(self, eval_noise):
        if self.eval_dataset == 'cifar10C':
            return self.get_cifar10c_attack_loader(eval_noise)
        elif self.eval_dataset == 'imagenetC':
            return self.get_imagenetC_attack_loader(eval_noise)
        else:
            raise ValueError("Attack dataset not implemented")

    def get_cifar10c_attack_loader(self, eval_noise):

        if eval_noise == 'adversarial':
            _, _, test_loader = self.get_data_loaders()
            return test_loader

        transform_cifar10c = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        images = np.load(os.path.join('data/CIFAR-10-C', eval_noise))
        labels = np.load('data/CIFAR-10-C/labels.npy')
        cifar10c_dataset = AttackDataset(data=images,labels=labels,transform=transform_cifar10c)
        cifar10c_attack_loader = DataLoader(cifar10c_dataset, batch_size=self.batch_size, shuffle=False)

        return cifar10c_attack_loader

    def get_cifar10_loaders(self):
        # Define transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Load datasets
        trainset = datasets.CIFAR10(root=self.root, train=True, download=True, transform=transform_train)
        validset = datasets.CIFAR10(root=self.root, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=self.root, train=False, download=True, transform=transform_test)

        return trainset, validset, testset
    
    def get_imagenetC_attack_loader(self, eval_noise, n_classes=10):
        transform_imagenetc = transforms.Compose([
            # TODO: might need different transforms for ImageNet
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        testset = ImageNetKaggle(os.path.join(self.root, "imagenet"), 'val', transform=transform_imagenetc)
        data_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

        return data_loader

        images = np.load(os.path.join('data/imagenet', eval_noise))
        labels = np.load('data/imagenet/labels.npy')
        imagenetc_dataset = AttackDataset(data=images, labels=labels, transform=transform_imagenetc)
        imagenetc_attack_loader = DataLoader(imagenetc_dataset, batch_size=self.batch_size, shuffle=False)
        return imagenetc_attack_loader

    def get_imagenet_loaders(self):
        # Define transforms for ImageNet
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Add normalization values specific to ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load datasets, assuming you have ImageNet dataset in the specified path
        trainset = ImageNetKaggle(os.path.join(self.root, "imagenet"), 'train', transform=transform_train)
        validset = ImageNetKaggle(os.path.join(self.root, "imagenet"), 'train', transform=transform_train)
        testset = ImageNetKaggle(os.path.join(self.root, "imagenet"), 'val', transform=transform_test)

        return trainset, validset, testset

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot_images_grid(images, labels, nrow=2, ncol=2, path='test.png'):
        fig, axs = plt.subplots(nrow, ncol, figsize=(8, 8))
        axs = axs.flatten()
        for img, lbl, ax in zip(images, labels, axs):
            img = img.permute(1, 2, 0)  # Change to HWC format for plotting
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Unnormalize
            img = img.numpy()
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f"Label: {lbl}")
            ax.axis('off')
        plt.savefig("test.png")

    # Initialize the DataLoaderFactory
    factory = DataLoaderFactory(root='./data', valid_size=0.1, train_dataset='imagenet', eval_dataset='imagenetC')

    train_loader, valid_loader, test_loader = factory.get_data_loaders()
    imagenetc_attack_loader = factory.get_attack_loader(eval_noise='defocus.npy')

    images_train, labels_train = next(iter(train_loader))
    plot_images_grid(images_train, labels_train, nrow=2, ncol=2, path="trainclean.png")
    
    images_valid, labels_valid = next(iter(valid_loader))
    plot_images_grid(images_valid, labels_valid, nrow=2, ncol=2, path="validclean.png")
    
    images_test, labels_test = next(iter(test_loader))
    plot_images_grid(images_test, labels_test, nrow=2, ncol=2, path="testclean.png")

    images_attack, labels_attack = next(iter(imagenetc_attack_loader))
    plot_images_grid(images_attack, labels_attack, nrow=2, ncol=2, path="cleanc.png")
