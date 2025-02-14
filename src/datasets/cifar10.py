from typing import *
import os
from filelock import FileLock
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10 as CIFAR10_PyTorch

class CIFAR10(CIFAR10_PyTorch):
    input_shape: Tuple[int] = (3, 32, 32)
    output_classes: int = 10
    mean: Tuple[float] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float] = (0.247, 0.243, 0.261)
    
    def __init__(self, root: Union[str, os.PathLike], train: bool = True):
        with FileLock(os.path.join(root, 'cifar10.lock')):
            _ = CIFAR10_PyTorch(root=root, train=True, download=True)
            _ = CIFAR10_PyTorch(root=root, train=False, download=True)
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)),
            download=False
        )