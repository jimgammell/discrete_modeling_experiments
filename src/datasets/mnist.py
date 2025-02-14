from typing import *
import os
from filelock import FileLock
import torch
from torchvision import transforms
from torchvision.datasets import MNIST as MNIST_PyTorch

class MNIST(MNIST_PyTorch):
    input_shape = (1, 32, 32)
    output_classes = 10
    mean = (0.1307,)
    std = (0.3081,)
    
    def __init__(self, root: Union[str, os.PathLike], train: bool = True):
        with FileLock(os.path.join(root, 'mnist.lock')):
            _ = MNIST_PyTorch(root=root, train=True, download=True)
            _ = MNIST_PyTorch(root=root, train=False, download=True)
        super().__init__(
            root=root,
            train=train,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
                transforms.Resize(size=(32, 32))
            ]),
            target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)),
            download=False
        )