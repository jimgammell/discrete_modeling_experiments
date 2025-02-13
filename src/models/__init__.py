

def load(model_name: str, *args, **kwargs):
    if model_name == 'ternary-mlp':
        from .architectures.ternary_mlp import TernaryMLP
        model = TernaryMLP(*args, **kwargs)
    elif model_name == 'ternary-lenet5':
        from .architectures.ternary_lenet5 import TernaryLeNet5
        model = TernaryLeNet5(*args, **kwargs)
    else:
        assert False
    return model