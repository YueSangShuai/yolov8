import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function, Variable
from torch.cuda.amp import autocast



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


from .quantization import ClippedLinearQuantization,LearnedClippedLinearQuantization,dorefa_quantize_param

class DoReFaQuantV2(nn.Module):
    def __init__(self, k=8):
        super(DoReFaQuantV2, self).__init__()
        self.k = k

    def forward(self, x):
        return dorefa_quantize_param(x, self.k)

    def extra_repr(self):
        return f'k={self.k}'



class Conv_with_bitwidthV2(nn.Module):
    
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, bitwidth=8):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.PACT = LearnedClippedLinearQuantization(num_bits=bitwidth,init_act_clip_val=10.0)  # 激活量化
        self.quantize = DoReFaQuantV2(k=bitwidth)  # 权重量化
        self.bitwidth = bitwidth
    @autocast(True)
    def forward(self, x):
        """Apply quantization, convolution, batch normalization, and activation to input tensor."""
        vhat = self.quantize(self.conv.weight)
        x = F.conv2d(x, vhat, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        x = self.bn(x)
        x = self.PACT(x)
        return x

    def forward_fuse(self, x):
        """Perform convolution and activation without batch normalization."""
        vhat = self.quantize(self.conv.weight)
        x = F.conv2d(x, vhat, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        x = self.PACT(x)
        return x



class Linear_with_bitwidthV2(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bitwidth=8):
        super(Linear_with_bitwidthV2, self).__init__(in_features, out_features, bias)
        self.quantize = DoReFaQuantV2(k=bitwidth)  # 权重量化
        self.bitwidth = bitwidth

    @autocast(True)
    def forward(self, x):
        vhat = self.quantize(self.weight)
        y = F.linear(x, vhat, self.bias)
        return y



class C2f_with_bitwidthV2(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,bitwidth=8):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv_with_bitwidthV2(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv_with_bitwidthV2((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_with_bitwidthV2(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))    


class Bottleneck_with_bitwidthV2(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_with_bitwidthV2(c1, c_, k[0], 1)
        self.cv2 = Conv_with_bitwidthV2(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
    
class Classify_with_bitwidthV2(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):

         """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
         padding, and groups.
         """
         super().__init__()
         c_ = 1280  # efficientnet_b0 size
         self.conv = Conv_with_bitwidthV2(c1, c_, k, s, p, g)
        #  self.bn = nn.BatchNorm1d(c_)
         self.bn = nn.BatchNorm2d(c_)
         self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
         self.drop = nn.Dropout(p=0.0, inplace=True)
         self.linear = nn.Linear(c_, c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)

        x = self.linear((self.drop(self.pool((self.conv(x))).flatten(1))))
        return x if self.training else x.softmax(1)