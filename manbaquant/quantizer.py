from re import U
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math
from .observers.hist_observers import PercentileObserver, KLObserver, MSEObserver
from .observers.minmax_observers import MinMaxObserver

CLIPMIN = 1e-5

class ClampSte(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,min_,max_):
        return x.clamp(min_,max_)
    
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output.clone(),None,None

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
        rescale=False,
        rescale_limit=False,
        has_batch_dim = False,
        is_weight=False,
        observe="minmax",
        percent = 0.999999,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point or self.symmetric:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        self.rescale = rescale # for channel-rescale
        self.rescale_limit = rescale_limit

        init_value = 4.0  # inti value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0] * math.ceil(shape[1] / group_size))
                self.deficiency = shape[-1] % group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric  # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
        
        if rescale:
            if rescale_limit:
                self.rescale_param = nn.Parameter(torch.zeros(dim1,1) )
            else:
                self.rescale_param = nn.Parameter(torch.ones(dim1,1) )

        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size
        
        self.has_batch_dim = has_batch_dim
        self.is_observing = False
        self.is_dynamic_quant = True
        granularity = 'dim{}'.format(per_channel_axes[0]) if len(per_channel_axes) > 0 else 'tensor'
        
        if observe == "percentile":
            self.observer = PercentileObserver(percent=0.999999,granularity=granularity)
        else:
            self.observer = MinMaxObserver(granularity=granularity)
 
        self.observered = False
        
        self.is_weight = is_weight

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]

        if self.rescale:
            rescale_param = self.rescale_param
            if self.rescale_limit:
                rescale_param = 0.5 + F.sigmoid(rescale_param)
            if len(rescale_param.shape) == 2 and len(x_dequant.shape)==3:
                rescale_param = rescale_param.unsqueeze(-1)
            x_dequant = x_dequant*rescale_param.to(x_dequant.device)
        return x_dequant

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
        
        if self.is_weight:#权重量化，没有observe过程
            if True:#not self.is_dynamic_quant:
                if  self.is_observing:
                    return x
                if self.observer is not None:
                    self.observer.update(x)
                    xmin,xmax = self.observer.cal_min_max()
                    self.assymmetric_cal_scale(xmin,xmax)
                    self.scale = self.expand_scale_shape_2_x(x, self.scale)
                    self.round_zero_point = self.expand_scale_shape_2_x(x, self.round_zero_point)
                    self.observer = None
                x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
                return x_dequant.type_as(x)
            # else:
            #     if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            #         self.per_token_dynamic_calibration(x)
            #     else:
            #         self.dynamic_per_tensor_calibration(x)
            #     x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
            #     return x_dequant
        else:#激活量化
            if not self.is_dynamic_quant:
                if self.is_observing:
                    self.observer.update(x)
                    return x.type_as(x)
                else:
                    if not self.observered:
                        xmin,xmax = self.observer.cal_min_max()
                        self.assymmetric_cal_scale(xmin,xmax)
                        self.scale = self.expand_scale_shape_2_x(x, self.scale)
                        self.round_zero_point = self.expand_scale_shape_2_x(x, self.round_zero_point)
                        self.observered = True
                        self.observer = None
                    x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
                    return x_dequant.type_as(x)
                    
            else:
                if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
                    self.per_token_dynamic_calibration(x)
                else:
                    self.dynamic_per_tensor_calibration(x)

                x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
                return x_dequant.type_as(x)

    def expand_scale_shape_2_x(self, x, scale):
        if self.per_channel_axes:
            dim=self.per_channel_axes[0]
            for i in range(len(x.shape)):
                if i != dim:
                    scale = scale.unsqueeze(i)
        return scale

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1, self.group_size)
            else:
                pad_zeros = torch.zeros(
                    (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
                )
                x = torch.cat((x, pad_zeros), dim=1)
                x = x.reshape(-1, self.group_size)
        if self.dynamic_method == "per_channel":
            if len(self.per_channel_axes):
                assert len(self.per_channel_axes) == 1,"must be one"
                reduce_shape = list(range(x.dim()))
                reduce_shape.remove(self.per_channel_axes[0])
            else:
                reduce_shape = list(range(x.dim()-1))
        else:
            reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax = x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        self.xmin_tmp = xmin.detach()
        self.xmax_tmp = xmax.detach()
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(self.scale)
        else:
            dynamic_range = xmax - xmin
            scale = dynamic_range / (2**self.n_bits - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
    
    def MaxMin_except_first_dim(self,tensor,func):
        # 获取张量的维度数
        dims = list(range(1, tensor.dim()))
        # 逐步在每个维度上取最大值
        for dim in dims:
            tensor, _ = func(tensor, dim=dim, keepdim=True)
        return tensor
    
    def dynamic_per_tensor_calibration(self,x):
        if not self.has_batch_dim:
            xmin = x.min()
            xmax = x.max()
        else:
            shape = [1] * len(x.shape)
            shape[0] = -1
            xmin = self.MaxMin_except_first_dim(x,torch.min).view(shape)
            xmax = self.MaxMin_except_first_dim(x,torch.max).view(shape)
        if self.symmetric or self.disable_zero_point:
            self.symmetric_cal_scale(xmin,xmax)
        else:
            self.assymmetric_cal_scale(xmin,xmax)

    def symmetric_cal_scale(self,xmin,xmax):
        abs_max = torch.max(xmax.abs(), xmin.abs())
        scale = abs_max / (2 ** (self.n_bits - 1) - 1)
        self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        self.round_zero_point = None
        
    def assymmetric_cal_scale(self,xmin,xmax):
        dynamic_range = xmax - xmin
        scale = dynamic_range / (2**self.n_bits - 1)
        self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        zero_point = -(xmin) / (self.scale)
        self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
    
    def normal_quantize(self, x, scales: torch.Tensor, mig_cof: torch.Tensor):
        s = (scales / mig_cof).max()
        s = s / (2**self.n_bits - 1)
        self.scale = s
        # only support symmetric quantization
        self.round_zero_point = None
        
    def scale_frexp(self):
        k = 16
        m = (self.scale*(2**k)).round()
        self.scale = m*(2**(-k))
        
        return self.scale

    def register_scales_and_zeros(self):
        self.register_buffer("scales", self.scale)
        self.register_buffer("zeros", self.round_zero_point)
        del self.scale
        del self.round_zero_point
        
    def quant2int(self, x):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / self.scale)
        if self.round_zero_point is not None:
            x_int = x_int.add(self.round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        
        if self.group_size:
            x_int = x_int.reshape(dim1, dim2)
        return x_int
    
    def dequant(self, x_int):
        if self.group_size:
            assert len(x_int.shape) == 2, "only support linear layer now"
            dim1, dim2 = x_int.shape
            x_int = x_int.reshape(-1, self.group_size)
            
        x_dequant = x_int
        if self.round_zero_point is not None:
            x_dequant = x_dequant.sub(self.round_zero_point)
        x_dequant = x_dequant.mul(self.scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]

        if self.rescale:
            rescale_param = self.rescale_param
            if self.rescale_limit:
                rescale_param = F.sigmoid(rescale_param) + 0.5
            x_dequant = x_dequant*self.rescale_param
        return x_dequant



class ActQuantizer(nn.Module):
    def __init__(self):
        self.register_parameter("scale",torch.ones(1))
        self.register_buffer("calibed_enabled",torch.tensor([0],dtype=torch.uint8))
    
    # @property
    # def calib
    
    def forward(self,x):
        pass


if __name__ == "__main__":
    cfg = {"dynamic_method":"per_tensor","n_bits":8,"symmetric":True}
    weight = torch.randn(100,100)
    quantizer = UniformAffineQuantizer(**cfg)
    weight_quant = quantizer.forward(weight)
    diff = weight-weight_quant
    print(diff.sum())