

def load(model_name: str, *args, **kwargs):
    if model_name == 'binary-mlp':
        from .architectures.binary_mlp import BinaryMLP
        model = BinaryMLP(*args, **kwargs)
    else:
        assert False
    return model