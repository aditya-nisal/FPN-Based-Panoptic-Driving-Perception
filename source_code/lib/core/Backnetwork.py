import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the backbone network (ResNet50)
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = resnet.layer1
        self.layer3 = resnet.layer2
        self.layer4 = resnet.layer3
        self.layer5 = resnet.layer4
        self.features = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4, self.layer5])

    def forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        return [c1, c2, c3, c4, c5]

#
    def forward(self, c):
        p = [self.lateral_convs[i](c[i]) for i in range(len(c))]
        for i in range(len(c)-1, 0, -1):
            p[i-1] += F.interpolate(p[i], scale_factor=2, mode='nearest')
        p = [self.fpn_convs[i](p[i]) for i in range(len(c))]
        p = [self.upsamples[i](p[i]) for i in range(len(c)-1)]
        p.append(p[-1])
        return p
class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
                                            for i in range(len(in_channels))])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                                        for _ in range(len(in_channels))])
        self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=2**(len(in_channels)-i), mode='nearest')
                                         for i in range(len(in_channels))])

    def forward(self, c):
        p = [self.lateral_convs[i](c[i]) for i in range(len(c))]
        for i in range(len(c)-1, 0, -1):
            p[i-1] += F.interpolate(p[i], scale_factor=2, mode='nearest')
        p = [self.fpn_convs[i](p[i]) for i in range(len(c))]
        p = [self.upsamples[i](p[i]) for i in range(len(c)-1)]
        p.append(p[-1])
        return p

# Define the YOLOP network
class YOLOP(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOP, self).__init__()
        self.backbone = Backbone()
        self.fpn1 = FPN([256, 512, 1024, 2048], 256)
        self.fpn2 = FPN([256, 512, 1024, 2048], 256)
        self.fpn3 = FPN([256, 512, 1024, 2048], 256)
        self.num_classes = num_classes
        self.yolop_head1 = YOLOPHead(256, num_classes)
        self.yolop_head2 = YOLOPHead(256, num_classes)
        self.yolop_head3 = YOLOPHead(256, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c = self.backbone(x)
        p = self.fpn(c)
        x = self.yolop_head(p)
        return x

class YOLOPHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(YOLOPHead, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels*2)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        self.conv3 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels*2)
        
        self.conv4 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(in_channels)
        
        self.conv5 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(in_channels*2)
        
        self.conv6 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(in_channels)
        
        self.conv7 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(in_channels*2)
        
        self.conv8 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn8 = nn.BatchNorm2d(in_channels)
        
        self.conv9 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(in_channels*2)
        
        self.conv10 = nn.Conv2d(in_channels*2, self.num_classes + 5, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu(x)
        
        x = self.conv10(x)
        return x


