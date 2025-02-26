import abc, sys, torch, torch.distributed as dist, math
from typing import *
from torch.distributed import ReduceOp
from .quant_param import QuantParam


eps = torch.tensor(torch.finfo(torch.float32).eps)

POW_QUANTIZATION = False


def set_pow_quantization(value: bool):
    global POW_QUANTIZATION
    assert value in (True, False)
    POW_QUANTIZATION = value


def get_pow_quantization():
    global POW_QUANTIZATION
    return POW_QUANTIZATION


def analysis_dim(granularities: list or str or int): # type: ignore
    """解析granularity, 并翻译为具体的channel数值, -1表示per-tensor, list将提取具体的通道数组成list, dim开头提取其后的通道数.

    Args:
        granularities (str or list): tensor or dimx, or [dim0, dim1, ...]

    Returns:
        int or list: 通道id
    """
    ch_axis = None
    if isinstance(granularities, list):
        ch_axis = []
        for granularity in granularities:
            if isinstance(granularity, str):
                assert len(granularity) == 4 and granularity[:3] == "dim"
                ch_axis.append(int(granularity[3:]))
            elif isinstance(granularity, int):
                ch_axis.append(granularity)
            else:
                raise NotImplemented
        for ch in ch_axis:
            assert ch >= 0, "for stability"
    elif isinstance(granularities, int):
        if granularities == -1:
            return -1
        return [granularities]
    elif granularities == "tensor":
        ch_axis = -1
    elif granularities[:3] == "dim":
        ch_axis = [
            int(granularities[3:]),
        ]
    return ch_axis


class ObserverABC(abc.ABC, torch.nn.Module):
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, granularity="tensor", min_limit=None, max_limit=None):
        super().__init__()
        self._granularity = None
        self._ch_axis = analysis_dim(granularity)
        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)
        self.align_with_set: Set[ObserverABC] = set()
        self.granularity = granularity
        self.manager: ObserverABC = None
        self.dtype = None
        self.symmetric = True
        self.min_limit = min_limit
        self.max_limit = max_limit

    def align_with(self, *args: Iterable["ObserverABC"]):
        for arg in args:
            if arg is not self:
                self.align_with_set.add(arg)

    @property
    def granularity(self):
        return self._granularity

    @granularity.setter
    def granularity(self, value):
        self._granularity = value
        self._ch_axis = analysis_dim(value)
        self.clear()

    def clear(self):
        self.min_val.resize_(0).fill_(0)
        self.max_val.resize_(0).fill_(0)

    @property
    def observer_name(self):
        return type(self).__name__

    def _pre_load_state_dict_hook(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        min_val = state_dict.get(prefix + "min_val", None)
        max_val = state_dict.get(prefix + "max_val", None)
        if min_val is not None:
            self.min_val.resize_(min_val.shape).copy_(min_val)
        if max_val is not None:
            self.max_val.resize_(max_val.shape).copy_(max_val)

    @property
    def ch_axis(self):
        return self._ch_axis
    
    @ch_axis.setter
    def ch_axis(self, new_ch_axis):
        self._ch_axis = new_ch_axis

    @torch.no_grad()
    def calculate_scale_zero_point(self, dtype, symmetric=True):
        self.symmetric = symmetric
        min_val, max_val = self.cal_min_max()
        if len(self.align_with_set):
            for to_align in self.align_with_set:
                if to_align.observer_name == 'FixedObserver':
                    _min_val, _max_val = to_align.cal_min_max()
                    if (_min_val.numel() + _max_val.numel() ) < 2 :
                        continue
                
                to_align.symmetric = symmetric
                to_align.dtype = dtype
                _min_val, _max_val = to_align.cal_min_max()
                min_val = torch.min(min_val, _min_val)
                max_val = torch.max(max_val, _max_val)
                # except:
                    # logger.error("observer align error")

        assert min_val is not None and max_val is not None
        quant_min, quant_max = dtype.qmin, dtype.qmax
        device = min_val.device
        scale = torch.ones(min_val.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val.size(), dtype=torch.int32, device=device)

        if symmetric:
            scale = torch.max(
                torch.abs(min_val / quant_min),
                torch.abs(max_val / quant_max),
            )
            scale = torch.max(scale, eps.to(scale.device))
            if POW_QUANTIZATION:
                scale = 1 / 2 ** (torch.floor((-1) * torch.log2(scale)).clamp(1, 14))
        else:
            scale = ((max_val - min_val) / float(quant_max - quant_min)).abs()
            scale = torch.max(scale, eps.to(scale.device))
            if POW_QUANTIZATION:
                scale = 1 / 2 ** (torch.floor((-1) * torch.log2(scale)).clamp(1, 14))
            zero_point = quant_min - torch.round(min_val / scale).to(torch.int32)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    @torch.no_grad()
    def calculate_qparams(self, dtype, symmetric=True):
        from .quant_param import QuantParam
        if self.manager is not None:
            return self.manager.calculate_qparams(dtype, symmetric)
        scale, zero_point = self.calculate_scale_zero_point(dtype, symmetric)
        grans = ""
        if isinstance(self.granularity, list):
            for gran in self.granularity:
                grans += gran
                grans += ","
            grans = grans[:-1]
        else:
            grans = self.granularity
        quant_param = QuantParam(
            dtype=f"int{dtype.bitwidth}",
            scale=scale,
            zero_point=zero_point,
            granularity=grans,
        )
        frame = sys._getframe()
        pre_frame = frame.f_back
        file_name = pre_frame.f_code.co_filename
        file_no = pre_frame.f_lineno
        self.quant_param = quant_param
        return self.quant_param

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if self.max_val.device != x.device:
            self.to(x.device)
        if self.manager is not None:
            self.manager.update(x)
        else:
            if x.dim() == 1 and self.ch_axis != -1:
                x = x.reshape(-1, 1)
            self._update_(x)

    def cal_min_max(self):
        min_val, max_val = self._cal_min_max_()
        if self.min_limit is not None:
            min_val.clip_(min=self.min_limit)
            max_val.clip_(min=self.min_limit)
        if self.max_limit is not None:
            max_val.clip_(max=self.max_limit)
            max_val.clip_(max=self.max_limit)
        min_val = torch.min(min_val, torch.zeros_like(min_val))
        max_val = torch.max(max_val, torch.zeros_like(max_val))
        if dist.is_initialized():
            dist.all_reduce(min_val, op=ReduceOp.MIN)
            dist.all_reduce(max_val, op=ReduceOp.MAX)
        return min_val, max_val

    def forward(self, x: torch.Tensor):
        self.update(x)
        return x

    # ********************必须要实现的抽象方法************************#
    @abc.abstractmethod
    def _cal_min_max_(self):
        return self.min_val, self.max_val

    @abc.abstractmethod
    def _update_(self, tensor: torch.Tensor):
        pass

    # ************************END**********************************#
