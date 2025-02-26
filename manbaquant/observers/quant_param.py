import numpy as np
import torch
import logging
import torch.nn as nn
import sys

logger = logging.getLogger("xuzukang.quant_param")


class QuantParam:
    def __init__(
        self,
        dtype,
        scale,
        zero_point=None,
        granularity="tensor",
        ste=True,
        small_scale_thresh=1e-20,
        drop_prob=0,
    ):
        """
        uniform quantization parameter
        @dtype, 'int*' | 'uint*' | 'fp32', e.g., int8
        @granularity, 'tensor' or [] | 'dim0' or [0] | 'dim1' or [1] | 'dim0,dim1' or [0,1] and so on
        @scale, float or Tensor or ndarray
        @zero_point, float or Tensor or ndarray
        @ste, is use Straight-Through Estimator in quant_tensor
        """

        if dtype[:5] == "float":
            raise NotImplementedError
            # bits = int(dtype[5:])
            # self.bitwidth = bits
            # self.is_float = True
        else:
            self.is_float = False
            if not isinstance(scale, torch.Tensor):
                if isinstance(scale, (int, float)):
                    scale = torch.tensor(1) * scale
                elif isinstance(scale, (list, tuple, np.ndarray)):
                    scale = np.array(scale)
                    scale = torch.from_numpy(scale).float()
                else:
                    raise NotImplementedError
            if zero_point is None:
                zero_point = torch.zeros_like(scale)
            if not isinstance(zero_point, torch.Tensor):
                if isinstance(zero_point, (int, float)):
                    zero_point = torch.ones_like(scale) * zero_point
                elif isinstance(zero_point, (list, tuple, np.ndarray)):
                    zero_point = np.array(zero_point)
                    zero_point = torch.from_numpy(zero_point).float()
                else:
                    raise NotImplementedError
            scale[torch.isinf(scale) | torch.isnan(scale)] = 1e-3
            self.scale = scale.to(torch.float)
            # self.scale[self.scale==0]=small_sacle_thresh

            assert not torch.isnan(self.scale).any()
            assert not torch.isinf(self.scale).any()
            if (self.scale <= small_scale_thresh).any():
                try:
                    frame = sys._getframe()
                    pre_frame = frame.f_back
                    file_name = pre_frame.f_code.co_filename
                    file_no = pre_frame.f_lineno
                    logger.warning(
                        f"from '{file_name}:{file_no}'  : scale <= small_sacle_thresh, set them to small_sacle_thresh={small_scale_thresh}"
                    )
                except Exception as e:
                    logger.warning(
                        f"WARNING: scale <= small_sacle_thresh, set them to small_sacle_thresh={small_scale_thresh}"
                    )
                self.scale[self.scale <= small_scale_thresh] = small_scale_thresh
            self.zero_point = zero_point.float().to(self.scale.device)
            assert self.scale.numel() == self.zero_point.numel()

            if (zero_point == 0).all():
                self.asymmetric = False
            else:
                self.asymmetric = True

            self.dtype = dtype

            if dtype[:3] == "int":
                bits = int(dtype[3:])
                self.qmin = -(1 << (bits - 1))
                self.qmax = (1 << (bits - 1)) - 1
                self.bitwidth = bits
            elif dtype[:4] == "uint":
                bits = int(dtype[4:])
                self.qmin = 0
                self.qmax = (1 << bits) - 1
                self.bitwidth = bits
            else:
                raise NotImplementedError

            self.granularity = granularity  # use the setter

            self.ste = ste
            self.drop_prob = drop_prob

    @property
    def granularity(self):
        return self._granularity

    @granularity.setter
    def granularity(self, value):
        if isinstance(value, (tuple, list)):
            if len(value) == 0:
                value = "tensor"
            else:
                value = ",".join([f"dim{_}" for _ in value])
        self._granularity = value
        if value is None or value == "tensor":
            self.granularity_dims = []
        else:
            if isinstance(value, str):
                dims = value.split(",")
                self.granularity_dims = [int(_[3:]) for _ in dims]
            elif isinstance(value, int):
                self.granularity_dims = [value]
            assert self.scale.dim() == len(self.granularity_dims)

    def set_ste_training(self, drop_prob=0):
        self.ste = True
        self.scale = nn.Parameter(self.scale.clone())
        self.drop_prob = drop_prob

    def set_eval(self):
        self.ste = False
        self.scale = self.scale.data.clone()
        self.drop_prob = 0
        self.training = False

    def __eq__(self, other):
        """两个QuantParam相等的前提是它的scale和zeropoint以及granularity都相等
        Args:
            other (QuantParam): 另一个QuantParam对象
        """
        return (
            self.scale.cpu() == other.scale.cpu()
            and self.zero_point.cpu() == other.zero_point.cpu()
            and self.granularity == other.granularity
        )

    def get_shaped_scale_zero(self, tensor):
        if self.granularity == "tensor":
            scale = self.scale
            zero_point = self.zero_point.view(1, 1, 1, 1)
        elif self.granularity == "dim0" and tensor.dim() == 4:
            scale = self.scale.view(-1, 1, 1, 1)
            zero_point = self.zero_point.view(-1, 1, 1, 1)
        elif self.granularity == "dim0" and tensor.dim() == 2:
            scale = self.scale.view(-1, 1)
            zero_point = self.zero_point.view(-1, 1)
        elif self.granularity == "dim1" and tensor.dim() == 4:
            scale = self.scale.view(1, -1, 1, 1)
            zero_point = self.zero_point.view(1, -1, 1, 1)
        elif self.granularity == "dim2" and tensor.dim() == 4:
            scale = self.scale.view(1, 1, -1, 1)
            zero_point = self.zero_point.view(1, 1, -1, 1)
        elif len(self.granularity_dims) == 1:
            shape = [
                -1 if self.granularity_dims[0] == i else 1 for i in range(tensor.dim())
            ]
            scale = self.scale.view(*shape)
            zero_point = self.zero_point.view(shape)
        else:
            new_shape = [1] * tensor.dim()
            for i, s in zip(self.granularity_dims, self.scale.size()):
                new_shape[i] = s
            if 0 not in self.granularity_dims:
                new_shape[0] = -1  # for batch size
            scale = self.scale.view(*new_shape)
            zero_point = self.zero_point.view(*new_shape)
        scale = scale.to(tensor.device)
        zero_point = zero_point.to(tensor.device)
        return scale, zero_point

    def quant_tensor(
        self, tensor: torch.Tensor, simulate=True, memory_efficient=False, toint=False
    ):
        """
        quantize tensor (activation or weight)
        x_integer=clamp(round(x/scale)+zero_point)
        x_simulate=(x_integer-zero_point)*scale
        """
        if self.is_float:
            return tensor
        if tensor.device != self.scale.device:
            self.scale = self.scale.to(tensor.device)
            self.zero_point = self.zero_point.to(tensor.device)
        scale, zero_point = self.get_shaped_scale_zero(tensor)
        # +0.5 to align with hardware implementation
        if self.ste:
            x = tensor.div(scale)
            if memory_efficient:
                integer = x.round_()
                if self.asymmetric:
                    integer = integer.add_(zero_point)
                out = integer.clamp_(self.qmin, self.qmax)
                integer = x = None
                if simulate:
                    if self.asymmetric:
                        out = out.sub_(zero_point)
                    out = out.mul_(scale)
                    assert self.drop_prob == 0
                else:
                    if out.min() >= -pow(2, 7) and out.max() <= (pow(2, 7) - 1):
                        out = out.char()
                    elif out.min() >= -pow(2, 15) and out.max() <= (pow(2, 15) - 1):
                        out = out.short()
                    elif out.min() >= -pow(2, 31) and out.max() <= (pow(2, 31) - 1):
                        out = out.int()
                    else:
                        out = out.long()
            else:
                integer = x + (x.round() - x).detach()  # round
                if self.asymmetric:
                    integer = integer.add(zero_point)
                out = integer.clamp(self.qmin, self.qmax)
                if simulate:
                    if self.asymmetric:
                        out = out.sub(zero_point)
                    out = out.mul(scale)
                    if self.drop_prob:
                        out = torch.where(
                            torch.rand_like(out) > self.drop_prob, out, tensor
                        )
                else:
                    if (
                        self.qmin >= -pow(2, 15)
                        and self.qmax <= (pow(2, 15) - 1)
                        and toint
                    ):
                        out = out.int()
                    else:
                        out = out.long()
        else:
            integer = tensor.div(scale).add_(0.5).floor_()
            if self.asymmetric:
                integer.add_(zero_point)
            out = integer.clamp_(self.qmin, self.qmax)
            if simulate:
                if self.asymmetric:
                    out = out.sub_(zero_point)
                out = out.mul_(scale)
            else:
                out = out.long()
        out.quant_param = self
        return out

    def hardware_saturated_clip(self, tensor_q, k, k_dim=None):
        """
        tensor_q should be long tensor
        k should be a int tensor
        """
        assert isinstance(tensor_q, (torch.LongTensor,torch.IntTensor,torch.cuda.LongTensor,torch.cuda.IntTensor))
        if isinstance(k, (int,)):
            k = torch.tensor(k, device=tensor_q.device).long()
        assert isinstance(k, (torch.LongTensor, torch.cuda.LongTensor))
        assert k.numel() == 1, "Not Implement yet"
        if k == 0:
            o = tensor_q
        elif k > 0:
            # o = (tensor_q + (1 << (k - 1)).clamp_(0)) >> k
            o = (
                tensor_q
                + (
                    torch.tensor(1, dtype=torch.long)
                    << (k - torch.tensor(1, dtype=torch.long))
                ).clamp_(0)
            ) >> k
        else:
            raise NotImplementedError
        o = o.clamp_(self.qmin, self.qmax)
        return o

    def integer_to_float(self, tensor, memory_efficient=False):
        if tensor.device != self.scale.device:
            self.scale = self.scale.to(tensor.device)
            self.zero_point = self.zero_point.to(tensor.device)
        scale, zero_point = self.get_shaped_scale_zero(tensor)
        if memory_efficient:
            if self.asymmetric:
                tensor = tensor.sub_(zero_point)
            tensor = tensor.mul_(scale)
        else:
            if self.asymmetric:
                tensor = tensor.sub(zero_point)
            tensor = tensor.mul(scale)
        tensor.quant_param = self
        return tensor

    def expand_to_shape(self, shape):
        tensor = torch.ones(shape).to(self.scale.device)
        self.zero_point = self.zero_point.to(self.scale.device)
        if self.granularity == "tensor":
            return tensor * self.scale, tensor * self.zero_point
        else:
            assert (
                np.prod([shape[_] for _ in self.granularity_dims]) == self.scale.numel()
            )
            new_shape = [1] * len(shape)
            for dim in self.granularity_dims:
                new_shape[dim] = shape[dim]
            return tensor * self.scale.view(new_shape), tensor * self.zero_point.view(
                new_shape
            )

    def compress_granularity_dims(
        self,
    ):
        assert self.scale.dim() == self.zero_point.dim()
        granularity_dims = []
        reduce_dims = []
        for i in range(self.scale.dim()):
            s_max = self.scale.amax(dim=i, keepdim=True)
            zp_max = self.zero_point.amax(dim=i, keepdim=True)
            if (s_max - self.scale).max() > self.scale.max() / 2 ** self.bitwidth or (
                zp_max - self.zero_point
            ).max() > self.zero_point.max() / 2**self.bitwidth:
                granularity_dims.append(i)
            else:
                reduce_dims.append(i)
        if len(reduce_dims):
            self.scale = self.scale.amax(reduce_dims)
            self.zero_point = self.zero_point.amax(reduce_dims)
        if len(granularity_dims):
            self.granularity = ",".join([f"dim{i}" for i in granularity_dims])
        else:
            self.granularity = "tensor"
        # self.granularity_dims = granularity_dims
        assert self.scale.dim() == len(self.granularity_dims)
        return self

    def __str__(self):
        info = f"quant {self.dtype} scale={self.scale} zero_point={self.zero_point} {self.granularity}"
        return info

    def __repr__(self) -> str:
        return self.__str__()

    def get_hardware_params(self, prefix=""):
        param_dict = {}
        scale = self.scale.cpu().view(-1).detach().numpy()
        if scale.shape == ():
            scale = scale.reshape(1)
        zero_point = self.zero_point.cpu().view(-1).detach().numpy()
        if zero_point.shape == ():
            zero_point = zero_point.reshape(1)
        scale = [float(_) for _ in scale]
        zero_point = [float(_) for _ in zero_point]
        param_dict[f"{prefix}dtype"] = self.dtype
        param_dict[f"{prefix}scale"] = scale
        param_dict[f"{prefix}zero_point"] = zero_point
        param_dict[f"{prefix}granularity"] = self.granularity

        return param_dict