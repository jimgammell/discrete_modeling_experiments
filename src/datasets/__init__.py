from common import *

def load_dataset(name):
    if name == 'MNIST':
        from .mnist import MNIST
        train_dataset = MNIST(root=MNIST_DIR, train=True)
        test_dataset = MNIST(root=MNIST_DIR, train=False)
    else:
        assert False
    return train_dataset, test_dataset