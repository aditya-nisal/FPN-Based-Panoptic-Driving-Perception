import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOPHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(YOLOPHead, self).__init__()
        self.num_classes = num_classes

        # 1x1 conv to reduce channel dimension
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)

        # apply 3x3 convolution to capture contextual information
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)

        # output prediction for each scale
        self.prediction1 = nn.Conv2d(in_channels, (num_classes + 5) * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.prediction2 = nn.Conv2d(in_channels // 2, (num_classes + 5) * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.prediction3 = nn.Conv2d(in_channels // 4, (num_classes + 5) * 3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x1, x2, x3 = x

        # FPN feature map fusion
        x3 = self.conv1(x3)
        x3 = self.bn1(x3)
        x3 = self.relu1(x3)
        x3 = F.interpolate(x3, scale_factor=2, mode="nearest")
        x2 = torch.cat([x2, x3], dim=1)

        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x2 = F.interpolate(x2, scale_factor=2, mode="nearest")
        x1 = torch.cat([x1, x2], dim=1)

        # output prediction for each scale
        pred1 = self.prediction1(x1)
        pred2 = self.prediction2(x2)
        pred3 = self.prediction3(x3)

        return pred1, pred2, pred3
