from typing import *

from ._base_resnet import _BaseResNet, BasicBlock, Bottleneck

class TernaryResNet18(_BaseResNet):
    def __init__(self, input_shape: Tuple[int], output_classes: int):
        super().__init__(
            input_shape=input_shape,
            output_classes=output_classes,
            block=BasicBlock,
            layers=[2, 2, 2, 2]
        )
class TernaryResNet34(_BaseResNet):
    def __init__(self, input_shape: Tuple[int], output_classes: int):
        super().__init__(
            input_shape=input_shape,
            output_classes=output_classes,
            block=BasicBlock,
            layers=[3, 4, 6, 3]
        )
class TernaryResNet50(_BaseResNet):
    def __init__(self, input_shape: Tuple[int], output_classes: int):
        super().__init__(
            input_shape=input_shape,
            output_classes=output_classes,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            zero_init_residual=True
        )
class TernaryResNet101(_BaseResNet):
    def __init__(self, input_shape: Tuple[int], output_classes: int):
        super().__init__(
            input_shape=input_shape,
            output_classes=output_classes,
            block=Bottleneck,
            layers=[3, 4, 23, 3],
            zero_init_residual=True
        )
class TernaryResNet152(_BaseResNet):
    def __init__(self, input_shape: Tuple[int], output_classes: int):
        super().__init__(
            input_shape=input_shape,
            output_classes=output_classes,
            block=Bottleneck,
            layers=[3, 8, 36, 3],
            zero_init_residual=True
        )