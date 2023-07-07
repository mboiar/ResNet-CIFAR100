from typing import Optional, Dict

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.typing import ArrayLike
import torch
import torchvision

CLASSES = ('apples', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottles',
           'bowls', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'cans', 'castle', 'caterpillar', 'cattle', 'chair',
           'chimpanzee', 'clock', 'cloud', 'cockroach', 'computer keyboard', 'couch', 'crab', 'crocodile',
           'cups', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
           'kangaroo', 'lamp', 'lawn-mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple', 'motorcycle',
           'mountain', 'mouse', 'mushrooms', 'oak', 'oranges', 'orchids', 'otter', 'palm', 'pears', 'pickup truck',
           'pine', 'plain', 'plates', 'poppies', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
           'roses', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
           'streetcar', 'sunflowers', 'sweet peppers', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
           'train', 'trout', 'tulips', 'turtle', 'wardrobe', 'whale', 'willow', 'wolf', 'woman', 'worm')


class BasicBlock(nn.Module):
    """Basic Block of ResNet."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward Pass of Basic Block."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class ResNet(nn.Module):
    """Residual Neural Network."""

    def __init__(self, num_classes: int = 100, in_channels: int = 32):
        super().__init__()

        # Pre
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.02)

        # Basic Blocks
        self.conv2_x = nn.Sequential(*[
            BasicBlock(32, 32) for i in range(4)
        ])

        self.conv3_x = nn.Sequential(BasicBlock(
            32, 64, 2,
            nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64))
        ))

        self.conv4_x = nn.Sequential(BasicBlock(
            64, 128, 2,
            nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128))
        ))

        self.conv5_x = nn.Sequential(BasicBlock(
            128, 256, 2,
            nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256))
        ))

        # Post
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc_layer = nn.Linear(256, num_classes)

        # initialize weights

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        """Forward pass of ResNet."""
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x


def preprocess(img: Optional[ArrayLike] = None) -> Optional[torch.Tensor]:
    if img is None:
        return None
    if len(img.shape) == 3:
        img = img[None, :, :, :]
    img = torch.from_numpy(img.copy())
    img = img.float()
    # img = img.unsqueeze(0)
    p = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    return p(img)

def get_probs(net: nn.Module, img: Optional[ArrayLike]) -> Dict:
    if img is None:
        return {}
    val, ind = torch.topk(F.softmax(net(img).data, dim=1), k=5)
    val, ind = val.numpy(), ind.numpy()
    return {CLASSES[ind[0][i]]:val[0][i].item() for i in range(val.size)}