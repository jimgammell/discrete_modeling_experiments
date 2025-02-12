from typing import *
import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST as MNIST_PyTorch

class MNIST(MNIST_PyTorch):
    def __init__(self, root: Union[str, os.PathLike], train: bool = True):
        super().__init__(
            root=root,
            train=train,
            transform=transforms.Compose([
                transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Resize(size=(32, 32))
            ]),
            target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)),
            download=False
        )
        self.input_shape = (1, 32, 32)
        self.output_classes = 10