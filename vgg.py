# Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import tqdm
import zipfile
import re
from collections import defaultdict
from math import sqrt
import tqdm
import pickle
from typing import Tuple, Any
VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

GHOST_SAMPLES = 40
class WeirdBatchNorm1d(nn.Module):
    def __init__(self, *args, ghost_samples=0, **kwargs):
        super().__init__()
        self.ghost_samples = ghost_samples
        self.bn = nn.BatchNorm1d(*args, **kwargs)
    def forward(self, x):
        if self.training:
            y = self.bn(x[-self.ghost_samples:])
            mean = x[-self.ghost_samples:].mean(0, keepdim=True)#.detach() # experimental detach to disconnect ghost sample from model and remaining batch
            var = x[-self.ghost_samples:].var(0, keepdim=True, unbiased=False)#.detach() # experimental detach to disconnect ghost sample from model and remaining batch
            tmp = (x - mean) / torch.sqrt(var + 1e-5)
            if self.bn.bias is not None:
                tmp * self.bn.weight + self.bn.bias
            return tmp
        else:
            return self.bn(x)
class WeirdBatchNorm2d(nn.Module):
    def __init__(self, *args, ghost_samples=GHOST_SAMPLES, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(*args, **kwargs)
        self.ghost_samples = ghost_samples

    def forward(self, x):
        if self.training:
            y = self.bn(x[-self.ghost_samples:])
            mean = x[-self.ghost_samples:].mean((0,2,3), keepdim=True)#.detach() # experimental detach to disconnect ghost sample from model and remaining batch
            var = x[-self.ghost_samples:].var((0,2,3), keepdim=True, unbiased=False)#.detach() # experimental detach to disconnect ghost sample from model and remaining batch
            tmp = (x - mean) / torch.sqrt(var + 1e-5)
            # print(tmp.shape, self.bn.weight.shape, self.bn.bias.shape)
            if self.bn.bias is not None:
                tmp = tmp * self.bn.weight.reshape(1, -1, 1, 1) + self.bn.bias.reshape(1, -1, 1, 1)
            return tmp
        else:
            return self.bn(x)

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])

        self.fcs = nn.Sequential(
            nn.Linear(512, 4096, bias=False),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes, bias=False),
        )
    
    def to_sequential(self):
        return nn.Sequential(*self.conv_layers, nn.Flatten(), *self.fcs)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        bias=False,
                    ),
                    WeirdBatchNorm2d(x, ghost_samples=GHOST_SAMPLES, affine=False, momentum=0.1),
                    # nn.LayerNorm(x),
                    # nn.GroupNorm(32, x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


class ResNet(torchvision.models.resnet.ResNet):
    """ResNet variant that knows about ghost samples during batch normalization and that can be converted to a torch.nn.Sequential model""" 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, norm_layer=WeirdBatchNorm2d, **kwargs)
        self.fc.bias = None
    def to_sequential(self):
        return nn.Sequential(
            self.conv1,
            self.bn1, 
            self.relu,
            self.maxpool,
            *self.layer1,
            *self.layer2,
            *self.layer3,
            *self.layer4,
            self.avgpool,
            nn.Flatten(),
            self.fc)

def resnet50(**kwargs):
    """Factory for resnet 50 models"""
    return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)

def wide_resnet50(**kwargs):
    return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], width_per_group=128, **kwargs)