# This code is adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional, Tuple

PADDING_MODE = 'reflect' # {'zeros', 'reflect', 'replicate', 'circular'}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, padding_mode = PADDING_MODE, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class WideResNetDec(nn.Module):
    def __init__(
        self,
        uplayers: List[int],
        skip_layers: List[int],
        pool: bool = False,
        skip_connection: bool = False,
        in_channels: int = 2048,
        num_classes: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64*2,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        final_activation: str = 'sigmoid',
        mean: list = [0.485, 0.456, 0.406], 
        std: list = [0.229, 0.224, 0.225] 
    ) -> None:
        super(WideResNetDec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.pool = pool
        self.mean = mean
        self.std = std
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [True, True, True]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.in_channels = in_channels
        self.upsample = nn.Upsample(scale_factor = 2, mode ='bicubic', align_corners = True)
        
        self.nin = nn.Sequential(
            conv1x1(self.in_channels, self.in_channels//2),
            nn.ReLU(inplace=True),
            conv1x1(self.in_channels//2, self.in_channels//2),
            nn.ReLU(inplace=True),
        )
        block  =  Bottleneck

        self.skip_layers = skip_layers
        if skip_connection==True:
            self.up_layers_channel_init = 128

            if 2 in self.skip_layers:
                inplanes = self.in_channels//2+self.in_channels//2
            else:
                inplanes = self.in_channels//2
            planes = self.up_layers_channel_init
            self.identity1 = nn.Identity()
            self.uplayer1 = self._make_layer(block, inplanes, planes, uplayers[0], stride=1, dilate=1)

            if 1 in self.skip_layers:
                inplanes = self.up_layers_channel_init* block.expansion + self.in_channels//4
            else:
                inplanes = self.up_layers_channel_init* block.expansion
            planes = self.up_layers_channel_init//2
            self.identity2 = nn.Identity()
            self.uplayer2 = self._make_layer(block, inplanes, planes, uplayers[1], stride=1, dilate=1)

            if 0 in self.skip_layers:
                inplanes = self.up_layers_channel_init//2* block.expansion + self.in_channels//8
            else:
                inplanes = self.up_layers_channel_init//2* block.expansion
            planes = self.up_layers_channel_init//4
            self.identity3 = nn.Identity()
            self.uplayer3 = self._make_layer(block, inplanes, planes, uplayers[2], stride=1, dilate=1)

            inplanes = self.up_layers_channel_init//4 
            inplanes = inplanes*block.expansion
            planes = self.up_layers_channel_init//8
            self.uplayer4 = self._make_layer(block, inplanes, planes, uplayers[3], stride=1, dilate=1)
        else:
            self.inplanes = self.in_channels
            self.up_layers_channel_init = self.in_channels//2

            inplanes = self.in_channels
            planes = self.up_layers_channel_init
            self.uplayer1 = self._make_layer(block, inplanes, planes, uplayers[0], stride=1, dilate=1)

            inplanes = self.up_layers_channel_init
            inplanes = inplanes*block.expansion
            planes = self.up_layers_channel_init//2
            self.uplayer2 = self._make_layer(block, inplanes, planes, uplayers[1], stride=1, dilate=1)

            inplanes = self.up_layers_channel_init//2
            inplanes = inplanes*block.expansion
            planes = self.up_layers_channel_init//4
            self.uplayer3 = self._make_layer(block, inplanes, planes, uplayers[2], stride=1, dilate=1)

            inplanes = self.up_layers_channel_init//4
            inplanes = inplanes*block.expansion
            planes = self.up_layers_channel_init//8
            self.uplayer4 = self._make_layer(block, inplanes, planes, uplayers[3], stride=1, dilate=1)

        self.conv1 = nn.Conv2d(self.up_layers_channel_init//8*block.expansion, self.up_layers_channel_init//16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.up_layers_channel_init//16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.up_layers_channel_init//16, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        if final_activation =='sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation =='softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation =='relu':
            self.final_activation = nn.ReLU(inplace=True)
        elif final_activation == 'leaky':
            self.final_activation = nn.LeakyReLU(inplace=True)
        else:
            self.final_activation = nn.Identity()

        self.normalize = T.Normalize(self.mean, self.std)
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

       # Zero-initialize the last BN in each residual branch,
       # so that the residual branch starts with zeros, and each residual block behaves like an identity.
       # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block: Type[Bottleneck], inplanes: int, planes: int, blocks: int,
                    stride: int = 1, dilate: int = 1, transpose: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        resample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            if transpose:
                resample = nn.Sequential(
                    conv1x1transpose(inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                resample = nn.Sequential(
                  conv1x1(inplanes, planes * block.expansion, stride),
                  norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(inplanes, planes, stride, resample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        for _ in range(1, blocks):
            layers.append(block(planes*block.expansion, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        if isinstance(x, tuple):
            x1, x2, x3, x4 = x

            x = self.nin(x4)

            x = self.upsample(x) #16x16
            if 2 in self.skip_layers:
                x = torch.cat((x3, x), 1)
            else:
                pass
            x = self.identity1(x)
            x = self.uplayer1(x)
            x = self.upsample(x) #32x32
            if 1 in self.skip_layers:
                x = torch.cat((x2, x), 1)
            else:
                pass
            x = self.identity2(x)
            x = self.uplayer2(x)
            x = self.upsample(x) #64x64
            if 0 in self.skip_layers:
                x = torch.cat((x1, x), 1)
            else:
                pass
            x = self.identity3(x)
            x = self.uplayer3(x)
            x = self.upsample(x) #128x128
            x = self.uplayer4(x)
        else:
            x = self.nin(x)

            x = self.upsample(x) #16x16
            x = self.uplayer1(x)
            x = self.upsample(x) #32x32
            x = self.uplayer2(x)
            x = self.upsample(x) #64x64
            x = self.uplayer3(x)
            x = self.upsample(x) #128x128
            x = self.uplayer4(x)

        x = self.upsample(x) #256x256
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.final_activation(x)
        if self.num_classes==3:
            x = self.normalize(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x) 

def wide_resnet50_2_dec(**kwargs: Any) -> WideResNetDec:
    return WideResNetDec([3,6,4,3], **kwargs)
