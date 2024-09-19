from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    # Smarter, conter gradiant complexity prblm by smoothing it thanks to connections skipping
    def __init__(self, c_size) -> None:
        super(ResidualConvBlock, self).__init__()
        self.conv1_1 = nn.Conv2d(c_size, c_size, kernel_size=(3, 1), padding=(1, 0))
        self.conv1_2 = nn.Conv2d(c_size, c_size, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(c_size)
        self.conv2_1 = nn.Conv2d(c_size, c_size, kernel_size=(3, 1), padding=(1, 0))
        self.conv2_2 = nn.Conv2d(c_size, c_size, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(c_size)

    def forward(self, x):
        # Double conv layer
        out = self.conv1_2(self.conv1_1(F.leaky_relu(self.bn1(x))))
        out = self.conv2_2(self.conv2_1(F.leaky_relu(self.bn2(out))))

        return out + x


class ResidualUpLayer(nn.Module):
    # Keep the residuality even when downsizing, upgrade compare to a classic ResNet
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            input_size, input_size * 2, padding=1, kernel_size=3, stride=2
        )
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgPool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = self.conv(x)
        x = torch.cat((self.maxPool(x), self.avgPool(x)), dim=1)
        return x + residual


class ResNet(nn.Module):
    def __init__(self, input_size: Union[int, Tuple[int]], nbr_classes: int) -> None:
        super(ResNet, self).__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        # Divide images
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1_1 = ResidualConvBlock(64)
        self.layer1_2 = ResidualConvBlock(64)
        # self.layer1_3 = ResidualConvBlock(64)

        self.conv2 = ResidualUpLayer(64)
        self.layer2_1 = ResidualConvBlock(128)
        self.layer2_2 = ResidualConvBlock(128)
        # self.layer2_3 = ResidualConvBlock(128)
        # self.layer2_4 = ResidualConvBlock(128)

        self.conv3 = ResidualUpLayer(128)
        self.layer3_1 = ResidualConvBlock(256)
        self.layer3_2 = ResidualConvBlock(256)
        self.layer3_3 = ResidualConvBlock(256)
        # self.layer3_4 = ResidualConvBlock(256)
        # self.layer3_5 = ResidualConvBlock(256)
        # self.layer3_6 = ResidualConvBlock(256)
        # self.layer3_7 = ResidualConvBlock(256)

        self.conv4 = ResidualUpLayer(256)
        self.layer4_1 = ResidualConvBlock(512)
        self.layer4_2 = ResidualConvBlock(512)
        # self.layer4_3 = ResidualConvBlock(512)
        # self.layer4_4 = ResidualConvBlock(512)
        # self.layer4_5 = ResidualConvBlock(512)

        # 2 Linear ouput layers
        print((input_size[0] // 32 + (0 if input_size[0] % 32 == 0 else 1)))
        print((input_size[1] // 32 + (0 if input_size[1] % 32 == 0 else 1)))
        self.fc1 = nn.Linear(
            512
            * (input_size[0] // 32 + (0 if input_size[0] % 32 == 0 else 1))
            * (input_size[1] // 32 + (0 if input_size[1] % 32 == 0 else 1)),
            1024,
        )
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, nbr_classes)

    def forward(self, x):
        # ResNet blocks
        x = self.conv0(x)
        x = self.maxPool(x)

        x = self.layer1_1(x)
        x = self.layer1_2(x)
        # x = self.layer1_3(x)

        x = self.conv2(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        # x = self.layer2_3(x)
        # x = self.layer2_4(x)

        x = self.conv3(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        # x = self.layer3_4(x)
        # x = self.layer3_5(x)
        # x = self.layer3_6(x)
        # x = self.layer3_7(x)

        x = self.conv4(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)
        # x = self.layer4_3(x)
        # x = self.layer4_4(x)
        # x = self.layer4_5(x)

        # Flatten dynamicly
        x = x.view(x.size(0), -1)

        # Output network
        x = F.dropout(F.leaky_relu(self.fc1(x)), 0.4)
        x = F.dropout(F.leaky_relu(self.fc2(x)), 0.4)
        x = self.fc3(x)
        return x
