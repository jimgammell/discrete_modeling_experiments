# Adapted from the TorchVision implementation

from typing import *
import torch
from torch import nn

from ...local_reparameterized_modules import TernaryConv2d, TernaryLinear

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> TernaryConv2d:
    return TernaryConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        scale_and_shift=False,
        dilation=dilation
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> TernaryConv2d:
    return TernaryConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        scale_and_shift=False
    )

class BasicBlock(nn.Module):
    expansion: int = 1
    
    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        assert groups == 1
        assert base_width == 64
        assert dilation == 1
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        if self.downsample is not None:
            x_skip = self.downsample(x_skip)
        x_resid = x
        x_resid = self.conv1(x_resid)
        x_resid = self.bn1(x_resid)
        x_resid = self.relu(x_resid)
        x_resid = self.conv2(x_resid)
        x_resid = self.bn2(x_resid)
        out = x_skip + x_resid
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion: int = 4
    
    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes*base_width/64)*groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        if self.downsample is not None:
            x_skip = self.downsample(x_skip)
        x_resid = x
        x_resid = self.conv1(x_resid)
        x_resid = self.bn1(x_resid)
        x_resid = self.relu(x_resid)
        x_resid = self.conv2(x_resid)
        x_resid = self.bn2(x_resid)
        x_resid = self.relu(x_resid)
        x_resid = self.conv3(x_resid)
        x_resid = self.bn3(x_resid)
        out = x_skip + x_resid
        out = self.relu(out)
        return out

class _BaseResNet(nn.Module):
    def __init__(self,
        input_shape: Sequence[int],
        output_classes: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_shape[0], self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, output_classes, bias=False)
        
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(mod, TernaryConv2d):
                initial_weights = torch.full(mod.weight_dims, torch.nan, dtype=mod.logits_dtype, device=mod.device)
                nn.init.kaiming_normal_(initial_weights, mode='fan_out', nonlinearity='relu')
                mod.reset_params(initial_weights)
            elif isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        
        if zero_init_residual:
            for mod in self.modules():
                if isinstance(mod, Bottleneck) and mod.bn3.weight is not None:
                    nn.init.constant_(mod.bn3.weight, 0)
                elif isinstance(mod, BasicBlock) and mod.bn2.weight is not None:
                    nn.init.constant_(mod.bn2.weight, 0)
    
    def _make_layer(self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                norm_layer(planes*block.expansion)
            )
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer
                )
            )
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)