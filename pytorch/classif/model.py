import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvNet(nn.Module):
    # !! Work only with 224x224 images !!
    # Fast but not super smart
    def __init__(self, nbr_classes):
        super(ConvNet, self).__init__()

        self.conv1_1 = nn.Conv2d(
            3, 3, kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)
        )
        self.conv1_2 = nn.Conv2d(
            3, 16, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )

        self.conv2_1 = nn.Conv2d(
            16, 16, kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)
        )
        self.conv2_2 = nn.Conv2d(
            16, 32, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )

        self.conv3_1 = nn.Conv2d(
            32, 32, kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)
        )
        self.conv3_2 = nn.Conv2d(
            32, 64, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )

        self.conv4_1 = nn.Conv2d(
            64, 64, kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)
        )
        self.conv4_2 = nn.Conv2d(
            64, 128, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)
        )

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(128 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, nbr_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1_2(self.conv1_1(x))))
        x = F.relu(self.bn2(self.conv2_2(self.conv2_1(x))))
        x = F.relu(self.bn3(self.conv3_2(self.conv3_1(x))))
        x = F.relu(self.bn4(self.conv4_2(self.conv4_1(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


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
        out = self.conv1_2(self.conv1_1(F.relu(self.bn1(x))))
        out = self.conv2_2(self.conv2_1(F.relu(self.bn2(out))))

        return out + x


class ResidualUpLayer(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_size, input_size*2, padding=1, kernel_size=3, stride=2)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgPool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return torch.cat((self.maxPool(x), self.avgPool(x)), dim=1) + self.conv(x)


class ResNet(nn.Module):
    # !! Work only with 224x224 images !!

    def __init__(self, nbr_classes: int) -> None:
        super(ResNet, self).__init__()

        # Divide images
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2)

        # layers
        self.layer1_1 = ResidualConvBlock(64)
        self.layer1_2 = ResidualConvBlock(64)

        self.conv2 = ResidualUpLayer(64)
        self.layer2_1 = ResidualConvBlock(128)
        self.layer2_2 = ResidualConvBlock(128)

        self.conv3 = ResidualUpLayer(128)
        self.layer3_1 = ResidualConvBlock(256)
        self.layer3_2 = ResidualConvBlock(256)
        self.layer3_3 = ResidualConvBlock(256)

        self.conv4 = ResidualUpLayer(256)
        self.layer4_1 = ResidualConvBlock(512)
        self.layer4_2 = ResidualConvBlock(512)

        # 2 Linear ouput layers
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, nbr_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # ResNet blocks
        x = self.conv1(x)
        x = self.maxPool(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)

        x = self.conv2(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)

        x = self.conv3(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)

        x = self.conv4(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)

        # Flatten dynamicly
        x = x.view(x.size(0), -1)

        # Output network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
