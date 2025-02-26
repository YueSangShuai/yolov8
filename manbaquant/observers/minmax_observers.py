import torch
from ._register import ObserverRegister
from .observer_abc import *


@ObserverRegister.add("minmax", "Minmax", "MINMAX")
class MinMaxObserver(ObserverABC):
    def _cal_min_max_(self):
        return super()._cal_min_max_()

    def _update_(self, x: torch.Tensor):
        dims = tuple(range(x.dim()))
        if self.ch_axis != -1:
            dims = [dim for dim in dims if dim not in self.ch_axis]
        max_val = torch.amax(x, dim=dims, keepdim=False)
        min_val = torch.amin(x, dim=dims, keepdim=False)
        if max_val.dim() == 0 or min_val.dim() == 0:
            assert max_val.dim() == min_val.dim()
            max_val = max_val.reshape(-1)
            min_val = min_val.reshape(-1)
        if self.max_val.numel() == 0:
            self.max_val.resize_(max_val.shape).fill_(0)
        if self.min_val.numel() == 0:
            self.min_val.resize_(min_val.shape).fill_(0)
        self.max_val.data.copy_(torch.max(self.max_val, max_val))
        self.min_val.data.copy_(torch.min(self.min_val, min_val))


@ObserverRegister.add("ema", "EMA", "Ema", "emaminmax", "EmaMinmax")
class EMAMinMaxObserver(ObserverABC):
    def __init__(
        self,
        granularity="tensor",
        averaging_constant=0.05,
        min_limit=None,
        max_limit=None,
    ):
        super().__init__(granularity, min_limit, max_limit)
        self.averaging_constant = averaging_constant

    def _cal_min_max_(self):
        return super()._cal_min_max_()

    def _update_(self, x: torch.Tensor):
        dims = tuple(range(x.dim()))
        if self.ch_axis != -1:
            dims = [dim for dim in dims if dim not in self.ch_axis]
        max_val_cur = torch.amax(x, dim=dims, keepdim=False)
        min_val_cur = torch.amin(x, dim=dims, keepdim=False)
        if max_val_cur.dim() == 0 or min_val_cur.dim() == 0:
            max_val_cur = max_val_cur.reshape(-1)
            min_val_cur = min_val_cur.reshape(-1)
        if self.max_val.numel() == 0:
            self.max_val.resize_(max_val_cur.shape).copy_(max_val_cur)
        if self.min_val.numel() == 0:
            self.min_val.resize_(min_val_cur.shape).copy_(min_val_cur)

        self.max_val.copy_(
            self.max_val + self.averaging_constant * (max_val_cur - self.max_val)
        )
        self.min_val.copy_(
            self.min_val + self.averaging_constant * (min_val_cur - self.min_val)
        )


@ObserverRegister.add("fixed", "FIXED", "Fixed")
class FixedObserver(ObserverABC):
    def __init__(
        self,
        granularity="tensor",
        min=None,
        max=None,
        scale=None,
        zero_point=0,
        # dtype="int8",
    ):
        super().__init__(granularity)
        assert (min is not None and max is not None) or (
            scale is not None
        ), "did't have "
        # dtype = Dtype.convert_to_dtype(dtype)
        # if scale is not None:
        #     max = (dtype.qmax - zero_point) * scale
        #     min = (dtype.qmin - zero_point) * scale
        self.fixed_scale = False
        if scale is not None:
            self.fixed_scale = True
            scale = torch.tensor(scale)
            zp = torch.tensor(zero_point)
            if scale.dim() == 0:
                scale = scale.reshape(-1)
                zp = zp.reshape(-1)
            self.register_buffer("scale", scale)
            self.register_buffer("zero_point", zp)
        else:
            min = torch.tensor(min)
            max = torch.tensor(max)

            if min.dim() == 0:
                min = min.reshape(-1)
                max = max.reshape(-1)
            self.min_val.resize_(min.shape).copy_(min)
            self.max_val.resize_(max.shape).copy_(max)

    @torch.no_grad()
    def calculate_scale_zero_point(self, dtype, symmetric=True):
        if self.fixed_scale:
            return self.scale, self.zero_point
        else:
            return super().calculate_scale_zero_point(dtype, symmetric)

    def _cal_min_max_(self):
        return super()._cal_min_max_()

    def _update_(self, x):
        pass


@ObserverRegister.add("aciq", "ACIQ", "Aciq")  # TODO
class AciqObserver(ObserverABC):
    def __init__(
        self,
        granularity="tensor",
        min_limit=None,
        max_limit=None,
    ):
        super().__init__(granularity, min_limit, max_limit)
        self.element_num = 0

    def _cal_min_max_(self):
        return super()._cal_min_max_()

    def _update_(self, x: torch.Tensor):
        dims = tuple(range(x.dim()))

        if self.ch_axis != -1:
            dims = [dim for dim in dims if dim not in self.ch_axis]
            ele_num = 1
            for dim in dims:
                ele_num = ele_num * x.shape[dim]
            self.element_num += ele_num
        else:
            self.element_num += x.numel()

        max_val = torch.amax(x, dim=dims, keepdim=False)
        min_val = torch.amin(x, dim=dims, keepdim=False)

        if max_val.dim() == 0 or min_val.dim() == 0:
            assert max_val.dim() == min_val.dim()
            max_val = max_val.reshape(-1)
            min_val = min_val.reshape(-1)
        if self.max_val.numel() == 0:
            self.max_val.resize_(max_val.shape).fill_(0)
        if self.min_val.numel() == 0:
            self.min_val.resize_(min_val.shape).fill_(0)

        max_val = self.compute_aciq_gaussian_clip(
            torch.max(self.max_val, max_val), self.element_num
        )
        min_val = self.compute_aciq_gaussian_clip(
            torch.max(self.min_val, min_val), self.element_num
        )

        self.max_val.data.copy_(max_val)
        self.min_val.data.copy_(min_val)

    def compute_aciq_gaussian_clip(
        self,
        max_value: torch.Tensor,
        N: int,
        bit_width: int = 8,
        distribution="gaussian",
    ):
        """计算threshold
        max_value: 最大值
        N: 元素个数
        """
        if distribution == "gaussian":
            alpha_gaussian = [
                0,
                1.71063519,
                2.15159277,
                2.55913646,
                2.93620062,
                3.28691474,
                3.6151146,
                3.92403714,
            ]
            # 当8-bit量化时，α=3.9240371
            gaussian_const = (0.5 * 0.35) * (
                1 + math.sqrt(3.14159265358979323846 * math.log(4))
            )
            std = (max_value * 2 * gaussian_const) / math.sqrt(2 * math.log(N))
            return alpha_gaussian[bit_width - 1] * std

        elif distribution == "laplace":
            raise NotImplementedError  # TODO
        else:
            raise KeyError
