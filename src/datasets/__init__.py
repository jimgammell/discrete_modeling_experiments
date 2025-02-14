from common import *
from .data_module import DataModule

def load_dataset(name):
    if name == 'mnist':
        from .mnist import MNIST
        train_dataset = MNIST(root=MNIST_DIR, train=True)
        test_dataset = MNIST(root=MNIST_DIR, train=False)
    elif name == 'cifar10':
        from .cifar10 import CIFAR10
        train_dataset = CIFAR10(root=CIFAR10_DIR, train=True)
        test_dataset = CIFAR10(root=CIFAR10_DIR, train=False)
    else:
        assert False
    return train_dataset, test_dataset