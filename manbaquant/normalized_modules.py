import operator
import torch, torch.nn as nn, torch.nn.functional as F, torch.utils._pytree as pytree
from torch import Tensor
from typing import Optional
TORCH_VERSION = torch.__version__

class NormalizedFuncMeta(type):
    def __new__(
        cls, name: str, bases: tuple, attrs: dict
    ):  # 元类的new方法创造一个类并返回,就像类的new方法创建一个实例并返回一样,cls为元类的Mata
        assert "torch_fn" in attrs.keys(), ""
        if nn.Module not in bases:
            bases = (nn.Module,) + bases

        """ 不通用,不使用
        def init_method(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            for k, v in kwargs:
                setattr(self, k, v)
        """

        def forward(self, *args, **kwargs):
            return self.torch_fn(*args, **kwargs)

        if not attrs.get("forward", None):
            attrs["forward"] = forward
        return super().__new__(cls, name, bases, attrs)


class Add(metaclass=NormalizedFuncMeta):
    torch_fn = operator.add

class Clone(metaclass=NormalizedFuncMeta):
    @staticmethod
    def f(x):
        return x.clone()

    torch_fn = f
    
class Equal(metaclass=NormalizedFuncMeta):
    torch_fn = torch.eq

class Sub(metaclass=NormalizedFuncMeta):
    torch_fn = operator.sub

class Mul(metaclass=NormalizedFuncMeta):
    torch_fn = operator.mul

class Div(metaclass=NormalizedFuncMeta):
    torch_fn = operator.truediv

class FloorDiv(metaclass=NormalizedFuncMeta):
    torch_fn = torch.floor_divide


class Exp(metaclass=NormalizedFuncMeta):
    torch_fn = torch.exp

class Tanh(metaclass=NormalizedFuncMeta):
    torch_fn = torch.tanh

class Sqrt(metaclass=NormalizedFuncMeta):
    torch_fn = torch.sqrt

class Log(metaclass=NormalizedFuncMeta):
    torch_fn = torch.log

class Sin(metaclass=NormalizedFuncMeta):
    torch_fn = torch.sin

class Cos(metaclass=NormalizedFuncMeta):
    torch_fn = torch.cos
    
class Acos(metaclass=NormalizedFuncMeta):
    torch_fn = torch.acos
    
class Atan(metaclass=NormalizedFuncMeta):
    torch_fn = torch.atan

class MatMul(metaclass=NormalizedFuncMeta):
    torch_fn = torch.matmul

class Arange(metaclass=NormalizedFuncMeta):
    torch_fn = torch.arange

class Chunk(metaclass=NormalizedFuncMeta):
    torch_fn = torch.chunk

class Abs(metaclass=NormalizedFuncMeta):
    torch_fn = torch.abs
    
class NanToNum(metaclass=NormalizedFuncMeta):
    torch_fn = torch.nan_to_num

class To(metaclass=NormalizedFuncMeta):
    @staticmethod
    def f(x, aim):
        return x.to(aim)

    torch_fn = f
    
# class Float(metaclass=NormalizedFuncMeta):
#     @staticmethod
#     def f(x):
#         return x.float()

#     torch_fn = f

class LayerNormFp32(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, *args, **kwargs):
        return F.layer_norm(*args, **kwargs)


class Split(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.split(*args, **kwargs)
    
class Addcmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.addcmul(*args, **kwargs)
    
class Int(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.int32

    def forward(self, x):
        return x.to(self.dtype)

class ScalarSub(nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, x):
        return x - self.scalar

class Float(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32

    def forward(self, x):
        return x.to(self.dtype)


class Long(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.int64

    def forward(self, x):
        return x.to(self.dtype)


class Bool(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.bool

    def forward(self, x):
        return x.to(self.dtype)


class CPU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x.cpu(*args, **kwargs)


class CUDA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x.cuda(*args, **kwargs)


class Numpy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x.numpy(*args, **kwargs)
    
class Sum(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return torch.sum(x, *args, **kwargs)

class Shape(metaclass=NormalizedFuncMeta):
    @staticmethod
    def f(x):
        return x.shape

    torch_fn = f


class DType(metaclass=NormalizedFuncMeta):
    @staticmethod
    def f(x):
        return x.dtype

    torch_fn = f


class OnesLike(metaclass=NormalizedFuncMeta):
    torch_fn = torch.ones_like


class ZerosLike(metaclass=NormalizedFuncMeta):
    torch_fn = torch.zeros_like


class NanToNum(metaclass=NormalizedFuncMeta):
    torch_fn = torch.nan_to_num

class Repeat(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x.repeat(*args, **kwargs)


class Pow(nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent

    def forward(self, x):
        return torch.pow(x, self.exponent)


class Clip(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: torch.Tensor):
        return torch.clip(x, self.min, self.max)


class Type_as(nn.Module):
    def __init__(self):
        super().__init__()
        # self.tensor = tensor

    def forward(self, *args):
        return args[0].type_as(args[1])


class Expand(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.shape = size

    def forward(self, x):
        if isinstance(size, torch.Tensor):
            size = torch.Size([size[0], size[1]])
        output = x * torch.ones(size, dtype=x.type, device=x.device)
        return output


class ReduceMean(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.axes = dim
        self.keepdims = keepdim

    def forward(self, x):
        return torch.mean(x, self.axes, keepdim=self.keepdims)

class ReduceSum(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.axes = dim
        self.keepdims = keepdim

    def forward(self, x):
        return torch.sum(x, self.axes, keepdim=self.keepdims)

class TensorData(nn.Module):
    tensor: torch.Tensor

    def __init__(self, tensor):
        super().__init__()
        is_param = isinstance(tensor,nn.Parameter)
        if is_param:
            self.register_parameter("tensor", tensor)
        else:
            self.register_buffer("tensor", tensor)

    def forward(self):
        return self.tensor


class ReduceMax(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        if isinstance(dim, (float,int)):
            dim = [dim]
        self.axes = tuple(dim)
        self.keepdims = keepdim

    def forward(self, x):
        return torch.amax(x, self.axes, self.keepdims)

class ReduceMin(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        if len(dim) == 1:
            dim = [dim]
        self.axes = dim
        self.keepdims = keepdim

    def forward(self, x):
        return torch.amin(x, self.axes, self.keepdims)


class Concat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *tensors):
        return torch.concat(tensors, dim=self.dim)


class Stack(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, *tensors):
        return torch.stack(tensors, dim=self.dim)


def map_aggregate(fn, in_put):
    if isinstance(in_put, list):
        return [map_aggregate(fn, i) for i in in_put]
    elif isinstance(in_put, tuple):
        return tuple(map_aggregate(fn, i) for i in in_put)
    return fn(in_put)


def flatten(container: tuple):
    ret = list()
    map_aggregate(lambda t: ret.append(t), container)
    return ret


class Reshape(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, *shape):
        shape = flatten(shape)
        return torch.reshape(x, shape)


class Permute(nn.Module):
    def __init__(self, *perms, **kwargs):
        super().__init__()
        if len(perms) == 1 and isinstance(perms[0], (list,tuple)):
            self.dims = perms[0]
        else:
            self.dims = perms if perms else kwargs["dims"]

    def forward(self, x: torch.Tensor):
        return x.permute(self.dims).contiguous()


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        return x.transpose(self.dim0, self.dim1).contiguous()


class Expand(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, *size):
        return x.expand(size)


class Roll(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, *args, **kwargs):
        return torch.roll(*args, **kwargs)


class Unbind(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unbind(self.dim)


class Slice(nn.Module):
    def __init__(self, slice):
        self.slice = slice
        pass


class MatMul(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

class MulAdd(nn.Module):
    def __init__(self,add=0) -> None:
        super().__init__()
        self.add_val = add
    def forward(self, x, y):
        return torch.addcmul(self.add_val, x, y)
        # return torch.mul(x, self.mul_val) + self.add_val
        
        
class Squeeze(nn.Module):
    def __init__(self, axes=None):
        super().__init__()
        if axes is None:
            self.axes = axes
        else:
            if isinstance(axes, int):
                axes = [axes]
            self.axes = sorted(axes, reverse=True)

    def forward(self, x):
        rst = x
        if self.axes is None:
            self.axes = [_ for _ in range(x.ndim) if x.size(_) == 1]
            self.axes = sorted(self.axes, reverse=True)
        for a in self.axes:
            rst = rst.squeeze(a)
        return rst


class Unsqueeze(nn.Module):
    def __init__(self, axes=None):
        super().__init__()
        if isinstance(axes, int):
            axes = [axes]
        if axes != None:
            self.axes = sorted(axes, reverse=True)

    def forward(self, x):
        rst = x
        for axes_id in self.axes:
            rst = torch.unsqueeze(rst, dim=axes_id)
        return rst


class Getitem(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args):
        return operator.getitem(x, *args)


class Resize(metaclass=NormalizedFuncMeta):
    torch_fn = F.interpolate

    def __init__(
        self, size, scale_factor=None, resize_mode="nearest", align_corners=None
    ):
        super().__init__()
        self.out_size = size
        self.scale_factor = scale_factor
        self.resize_mode = resize_mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, self.out_size, self.resize_mode, self.align_corners)


class Pad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return F.pad(*args, **kwargs)


# 因为原版中的Upsample中的size是一个属性,但是很多情况下size是一个输入,因此重写Upsample算子
class Upsample(nn.Module):
    __constants__ = [
        "size",
        "scale_factor",
        "mode",
        "align_corners",
        "name",
        "recompute_scale_factor",
    ]
    name: str
    mode: str
    align_corners: Optional[bool]
    recompute_scale_factor: Optional[bool]

    def __init__(
        self,
        scale_factor=None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super(Upsample, self).__init__()
        self.name = type(self).__name__
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, input: Tensor, size=None) -> Tensor:
        return F.interpolate(
            input,
            size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )

    def extra_repr(self) -> str:
        info = ""
        if self.scale_factor is not None:
            info = info + "scale_factor=" + str(self.scale_factor)
        info += ", mode=" + self.mode
        return info


class DropOut(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, *args, **kwargs):
        return F.dropout(*args, **kwargs)


class Roll(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.roll(*args, **kwargs)


class ScalarAdd(nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, x):
        return x + self.scalar


class ScalarMul(nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, x):
        return x * self.scalar

class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, need_weights=True, average_attn_weights=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        if TORCH_VERSION > "1.11":
            return super().forward(
                query=query,
                key=key,
                value=value,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=self.need_weights,
                average_attn_weights=self.average_attn_weights,
            )
        else:
            return super().forward(
                query=query,
                key=key,
                value=value,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=self.need_weights,
            )


class Gridsample(nn.Module):
    name: str
    mode: str
    align_corners: Optional[bool]

    def __init__(
        self,
        mode: str = 'bilinear',
        padding_mode: str = "zeros",
        align_corners: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super(Gridsample, self).__init__()
        self.name = type(self).__name__
        self.mode = mode
        self.align_corners = align_corners if align_corners is not None else False
        self.padding_mode = padding_mode

    def forward(self, input: Tensor, grid=None) -> Tensor:
        return F.grid_sample(
            input,
            grid,
            self.mode,
            self.padding_mode,
            self.align_corners,
        )


if __name__ == "__main__":
    t = Exp()
    x = torch.randn(1, 3, 224, 224)
    t(x)
    pass
    Reshape()(torch.randn(1, 2, 3), 3, 2, 1)

    pass
