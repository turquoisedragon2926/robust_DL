import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from .components import AttackDataset

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
    
    def get_imagenetC_attack_loader(self, eval_noise):
        transform_imagenetc = transforms.Compose([
            # TODO: might need different transforms for ImageNet
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        images = np.load(os.path.join('data/ImageNet-C', eval_noise))
        labels = np.load('data/ImageNet-C/labels.npy')
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
        trainset = datasets.ImageNet(root=os.path.join(self.root, 'train'), split='train', download=True, transform=transform_train)
        validset = datasets.ImageNet(root=os.path.join(self.root, 'train'), split='train', download=True, transform=transform_train)
        testset = datasets.ImageNet(root=os.path.join(self.root, 'val'), split='val', download=True, transform=transform_test)

        return trainset, validset, testset
