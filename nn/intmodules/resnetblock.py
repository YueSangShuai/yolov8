import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function, Variable
from torch.cuda.amp import autocast


K = 4
global count
count = 0

def autopad(kernel, padding, dilation=1):
    if padding is None:
        padding = (kernel - 1) // 2 * dilation
    return padding

# k = 8
class ActFn(Function):
	@staticmethod
	def forward(ctx, x, alpha, k):
		ctx.save_for_backward(x, alpha)
		# y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
		y = torch.clamp(x, min = 0, max = alpha.item())
		scale = (2**k - 1) / alpha
		y_q = torch.round( y * scale) / scale
		return y_q

	@staticmethod
	def backward(ctx, dLdy_q):
		# Backward function, I borrowed code from
		# https://github.com/obilaniu/GradOverride/blob/master/functional.py
		# We get dL / dy_q as a gradient
		x, alpha, = ctx.saved_tensors
		# Weight gradient is only valid when [0, alpha]
		# Actual gradient for alpha,
		# By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
		# dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range 
		lower_bound      = x < 0
		upper_bound      = x > alpha
		# x_range       = 1.0-lower_bound-upper_bound
		x_range = ~(lower_bound|upper_bound)
		grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
		return dLdy_q * x_range.float(), grad_alpha, None

def quantize_k(r_i, k):
	scale = (2**k - 1)
	r_o = torch.round( scale * r_i ) / scale
	return r_o

class DoReFaQuant(Function):
	@staticmethod
	def forward(ctx, r_i, k):
		tanh = torch.tanh(r_i).float()
		# scale = 2**k - 1.
		# quantize_k = torch.round( scale * (tanh / 2*torch.abs(tanh).max() + 0.5 ) ) / scale
		r_o = 2*quantize_k( tanh / (2*torch.max(torch.abs(tanh)).detach()) + 0.5 , k) - 1
		# r_o = 2 * quantize_k - 1.
		return r_o

	@staticmethod
	def backward(ctx, dLdr_o):
		# due to STE, dr_o / d_r_i = 1 according to formula (5)
		return dLdr_o, None

class Conv_with_bitwidth(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bitwidth=8):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.quantize = DoReFaQuant.apply
        self.bitwidth = bitwidth

    @autocast(True)
    def forward(self, x):
        """Apply quantization, convolution, batch normalization, and activation to input tensor."""
        vhat = self.quantize(self.conv.weight, self.bitwidth)
        x = F.conv2d(x, vhat, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        x = self.bn(x)
        x = self.act(x)
        
        return x

    def forward_fuse(self, x):
        """Perform convolution and activation without batch normalization."""
        vhat = self.quantize(self.conv.weight, self.bitwidth)
        x = F.conv2d(x, vhat, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        x = self.act(x)
        return x

class Linear_with_bitwidth(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, bitwidth = 8):
        super(Linear_with_bitwidth, self).__init__(in_features, out_features, bias)
        self.quantize = DoReFaQuant.apply
        self.bitwidth = bitwidth
        
    @autocast(True)
    def forward(self, x):
        vhat = self.quantize(self.weight, self.bitwidth)
        y = F.linear(x, vhat, self.bias)

        return y

class ResNetBlock_with_bitwidth(nn.Module):
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
        self.shortcut = nn.Sequential(Conv_with_bitwidth(c1, c3, k=1, s=s, act=False, bitwidth = 8) if s != 1 or c1 != c3 else nn.Identity())

    def forward(self, x):      
        out = self.cv1(x)
        out = self.ActFn(out, self.alpha1, K)
        out = self.cv2(out)
        out = self.cv3(out)
        out = F.relu(out)
        out = out + self.shortcut(x)
        out = self.ActFn(out, self.alpha2, K)
        
        
        return out

class ResNetLayer_with_bitwidth(nn.Module):
    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        super().__init__()

        self.is_first = is_first
        self.alpha1 = nn.Parameter(torch.tensor(10.))
        self.ActFn = ActFn.apply

        if self.is_first:

            
            self.layer = nn.Sequential(
                Conv_with_bitwidth(c1, c2, k=7, s=2, p=3, act=True,bitwidth = 8), 
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock_with_bitwidth(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock_with_bitwidth(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)
        


    def forward(self, x):

        if self.is_first:
            x=self.layer(x)
            # print(self.ActFn(x, self.alpha1, K).shape)
            return self.ActFn(x, self.alpha1, K)
        else:
            return self.layer(x)

class Classify_with_bitwidth(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):

         """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
         padding, and groups.
         """
         super().__init__()
         c_ = 1280  # efficientnet_b0 size
         self.conv = Conv_with_bitwidth(c1, c_, k, s, p, g, bitwidth = 8)
         self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
         self.drop = nn.Dropout(p=0.0, inplace=True)
         self.linear = Linear_with_bitwidth(c_, c2, bitwidth = 8)

    def forward(self, x):
         """Performs a forward pass of the YOLO model on input image data."""
         if isinstance(x, list):
             x = torch.cat(x, 1)
         x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
         
         return x if self.training else x.softmax(1)
     
class v8ClassificationLoss_bitwidth:
    """Criterion class for computing training losses."""
    def __init__(self,model):
        self.model=model
    
    
    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        l2_alpha = 0.0
        lambda_alpha = 0.0002
        for name, param in self.model.named_parameters():
            if "alpha" in name:
                l2_alpha += torch.pow(param, 2)
                # print(f"{name}.requires_grad: {param.requires_grad}")
                # print(f"{name}.grad: {param.grad}")
                
        loss += lambda_alpha * l2_alpha
        
        loss_items = loss.detach()
        
        return loss, loss_items

