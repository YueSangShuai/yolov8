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

# k = 8import torch

class PACTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, bits):
        
        alpha_clamped = alpha.clamp(min=1e-3)  # 防止 alpha 过小
        ctx.save_for_backward(x, alpha)
        y = torch.clamp(x, min=-alpha.item(), max=alpha.item())
        scale = (2**bits - 1) / (alpha_clamped + 1e-8)
        y_q = torch.round(y * scale) / scale
        ctx.save_for_backward(x, alpha, y_q)  # 新增保存 y_q
        ctx.scale = scale  # 保存 scale 用于反向传播
        ctx.bits=bits
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        x, alpha, y_q = ctx.saved_tensors
        scale = ctx.scale
        
        # 输入梯度计算（带边界约束）
        grad_input = dLdy_q.clone()
        mask = (x >= -alpha) & (x <= alpha)
        grad_input[~mask] = 0  # 硬截断边界外的梯度

        # Alpha 梯度计算（带稳定化因子）
        grad_alpha = None
        if ctx.needs_input_grad[1]:
            # 量化误差项
            quant_error = (y_q - x) / (alpha + 1e-8)  # 归一化误差
            grad_quant = torch.sum(quant_error * (x.abs() >= alpha).float())
            
            # 边界梯度项
            boundary_grad = torch.sum(dLdy_q * (x.abs() >= alpha).float() * torch.sign(x))
            
            # 综合梯度（添加稳定化系数）
            grad_alpha = (grad_quant + boundary_grad) * (2**ctx.bits - 1) / (alpha**2 + 1e-6)
            
        return grad_input, grad_alpha, None


class PACT(nn.Module):
    def __init__(self, alpha=5.0, bits=8):  # 减小初始 alpha
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.bits = bits
        # 初始化建议：根据输入分布动态调整
        nn.init.normal_(self.alpha, mean=6.0, std=0.5)  # 高斯初始化


    def forward(self, x):
        return PACTFunction.apply(x, self.alpha, self.bits)

    def extra_repr(self):
        return f'alpha={self.alpha.item():.4f}, bits={self.bits}'


def quantize_k(r_i, k):
	scale = (2**k - 1)
	r_o = torch.round( scale * r_i ) / scale
	return r_o

class DoReFaQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_i, k):
        tanh = torch.tanh(r_i)
        max_abs = torch.max(torch.abs(tanh)) + 1e-8  # 防止除零
        norm = (tanh / (2 * max_abs)) + 0.5
        scale = 2**k - 1
        quantized = torch.round(norm * scale) / scale
        r_o = 2 * quantized - 1
        ctx.save_for_backward(r_i, tanh, max_abs)  # 保存中间变量
        ctx.scale = scale
        

        return r_o

    @staticmethod
    def backward(ctx, dLdr_o):

        return dLdr_o, None


class DoReFaQuant(nn.Module):
    def __init__(self, k=8):
        super(DoReFaQuant, self).__init__()
        self.k = k

    def forward(self, x):
        return DoReFaQuantFunction.apply(x, self.k)

    def extra_repr(self):
        return f'k={self.k}'



class Conv_with_bitwidth(nn.Module):
    
    default_act = nn.ReLU6()  # default activation
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, bitwidth=8,act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.PACT = PACT(bits=bitwidth)  # 激活量化
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.quantize = DoReFaQuant(k=bitwidth)  # 权重量化
        self.bitwidth = bitwidth
        
    @autocast(True)
    def forward(self, x):
        """Apply quantization, convolution, batch normalization, and activation to input tensor."""
        vhat = self.quantize(self.conv.weight)
        x = F.conv2d(x, vhat, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        x = self.bn(x)
        x=self.act(x)
        x = self.PACT(x)
        return x

    def forward_fuse(self, x):
        """Perform convolution and activation without batch normalization."""
        vhat = self.quantize(self.conv.weight)
        x = F.conv2d(x, vhat, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        # x=self.conv(x)
        x=self.act(x)
        x = self.PACT(x)
        return x


class Linear_with_bitwidth(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bitwidth=8):
        super(Linear_with_bitwidth, self).__init__(in_features, out_features, bias)
        self.quantize = DoReFaQuant(k=bitwidth)  # 权重量化
        self.bitwidth = bitwidth

    @autocast(True)
    def forward(self, x):
        vhat = self.quantize(self.weight)
        y = F.linear(x, vhat, self.bias)
        return y


class C2f_with_bitwidth(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,bitwidth=8):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv_with_bitwidth(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv_with_bitwidth((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_with_bitwidth(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        
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


class Bottleneck_with_bitwidth(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_with_bitwidth(c1, c_, k[0], 1)
        self.cv2 = Conv_with_bitwidth(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
    
class Classify_with_bitwidth(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):

         """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
         padding, and groups.
         """
         super().__init__()
         c_ = 1280  # efficientnet_b0 size
         self.conv = Conv_with_bitwidth(c1, c_, k, s, p, g)
        #  self.bn = nn.BatchNorm1d(c_)
         self.bn = nn.BatchNorm2d(c_)
         self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
         self.drop = nn.Dropout(p=0.0, inplace=True)
         self.linear = Linear_with_bitwidth(c_, c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)

        x = self.linear((self.drop(self.pool((self.conv(x))).flatten(1))))
        return x if self.training else x.softmax(1)