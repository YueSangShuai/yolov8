import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from module import ActFn, Conv_with_bitwidth, Linear_with_bitwidth
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
K = 2
print("Bit :", K)

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ResNetBlock_with_bitwidth(nn.Module):
    expansion = 1

    def __init__(self, c1, c2, s=1, e=4):
        super().__init__()
        c3 = e * c2
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.cv1 = Conv_with_bitwidth(c1, c2, k=1, s=1, act=True, bitwidth = K)
        self.alpha1 = nn.Parameter(torch.tensor(10.))
        self.cv2 = Conv_with_bitwidth(c2, c2, k=3, s=s, p=1, act=True,bitwidth = K)
        self.alpha2 = nn.Parameter(torch.tensor(10.))
        self.cv3 = Conv_with_bitwidth(c2, c3, k=1, act=False,bitwidth = K)
        self.ActFn = ActFn.apply
        self.shortcut = self.shortcut = nn.Sequential(Conv_with_bitwidth(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        
        out = self.cv1(x)
        out = self.ActFn(out, self.alpha1, K)
        out = self.cv2(x)
        out = self.cv3(x)
        out += self.shortcut(x)
        out = self.ActFn(out, self.alpha2, K)
        
        return out

class ResNetLayer_with_bitwidth(nn.Module):
    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        super().__init__()
        self.in_planes = 16

        self.is_first = is_first
        
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, bitwidth = 8)
        self.bn1 = nn.BatchNorm2d(16)
        self.alpha1 = nn.Parameter(torch.tensor(10.))
        self.ActFn = ActFn.apply
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # self.linear = nn.Linear(64, num_classes)
        self.linear = Linear(64, num_classes, bitwidth = 8)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.ActFn(self.bn1(self.conv1(x)), self.alpha1, K)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out


class Classify_with_bitwidth(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2):

         """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
         padding, and groups.
         """
         super().__init__()
         self.linear = Linear(64, c2, bitwidth = 8)

    def forward(self, x):
         """Performs a forward pass of the YOLO model on input image data."""
         if isinstance(x, list):
             x = torch.cat(x, 1)
         x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
         
         return x if self.training else x.softmax(1)


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3],num_classes=2)