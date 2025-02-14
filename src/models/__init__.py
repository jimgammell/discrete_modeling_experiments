def load(model_name: str, *args, **kwargs):
    if model_name == 'ternary-mlp':
        from .architectures.ternary_mlp import TernaryMLP
        model = TernaryMLP(*args, **kwargs)
    elif model_name == 'ternary-lenet5':
        from .architectures.ternary_lenet5 import TernaryLeNet5
        model = TernaryLeNet5(*args, **kwargs)
    elif model_name == 'ternary-resnet-18':
        from .architectures.ternary_resnet import TernaryResNet18
        model = TernaryResNet18(*args, **kwargs)
    elif model_name == 'ternary-resnet-34':
        from .architectures.ternary_resnet import TernaryResNet34
        model = TernaryResNet34(*args, **kwargs)
    elif model_name == 'ternary-resnet-50':
        from .architectures.ternary_resnet import TernaryResNet50
        model = TernaryResNet50(*args, **kwargs)
    elif model_name == 'ternary-resnet-101':
        from .architectures.ternary_resnet import TernaryResNet101
        model = TernaryResNet101(*args, **kwargs)
    elif model_name == 'ternary-resnet-152':
        from .architectures.ternary_resnet import TernaryResNet152
        model = TernaryResNet152(*args, **kwargs)
    else:
        assert False
    return model