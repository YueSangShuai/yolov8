import torch, numpy as np
from functools import partial
from copy import deepcopy
from typing import Union
from .quant_param import QuantParam
from ._register import ObserverRegister
from ._hist_manager import HistManager
from .observer_abc import ObserverABC, eps


@ObserverRegister.add("percent", "Percent", "PERCENT")
class PercentileObserver(ObserverABC):
    ch_shapes: list

    def __init__(
        self,
        granularity="tensor",
        hist_bin_num: int = 2048,
        percent: Union[float, list] = 1.0,
        percentile_mode: str = "line",
        min_limit = None,
        max_limit = None
    ):
        super().__init__(granularity,min_limit,max_limit)
        if isinstance(percent, float):
            self.left_percent = self.right_percent = percent
        else:
            self.left_percent, self.right_percent = percent
        self.percentile_mode = percentile_mode
        self.hist_manager = HistManager(num_bins=hist_bin_num)
        self.ch_shapes = 1

    def _update_(self, tensor: torch.Tensor):
        hist_manager = self.hist_manager
        if self.ch_axis == -1:
            x = tensor.contiguous().view(1, -1)
        elif isinstance(self.ch_axis, list):
            self.ch_shapes = [tensor.shape[i] for i in self.ch_axis]
            dims = list(range(tensor.dim()))  # self.ch_shapes =
            permute_dims = deepcopy(self.ch_axis)
            for dim in dims:
                if dim not in permute_dims:
                    permute_dims.append(dim)
            x = tensor.permute(permute_dims)
            x = x.reshape(int(np.prod(self.ch_shapes)), -1)  # (#channels, -1)
        else:
            raise NotImplementedError("ch axis must be int or list.")
        hist_manager.collect(data=x)

    def percentile(self):
        if self.left_percent >= 1.0 or self.right_percent >= 1.0:
            assert (
                self.percentile_mode == "line"
            ), "If percent is 1.0, must use line for no loss."
        min_clip_tensor, max_clip_tensor = self.hist_manager.percentile(
            left_percent=self.left_percent,
            right_percent=self.right_percent,
            mode=self.percentile_mode,
        )
        return min_clip_tensor, max_clip_tensor

    def _cal_min_max_(self):
        min_val, max_val = self.percentile()
        min_val = min_val.reshape(self.ch_shapes)
        max_val = max_val.reshape(self.ch_shapes)
        self.min_val.resize_(min_val.shape).copy_(min_val)
        self.max_val.resize_(max_val.shape).copy_(max_val)
        return self.min_val, self.max_val


@ObserverRegister.add("kl", "KL", "Kl")
class KLObserver(ObserverABC):
    ch_shapes: list

    def __init__(
        self,
        granularity="tensor",
        hist_bin_num: int = 2048,
        iter_times=1024,
        min_limit = None,
        max_limit = None,
    ):
        super().__init__(granularity,min_limit,max_limit)
        self.hist_manager = HistManager(num_bins=hist_bin_num)
        self.ch_shapes = 1
        self.iter_times = iter_times

    def _update_(self, tensor: torch.Tensor):
        hist_manager = self.hist_manager
        if self.ch_axis == -1:
            x = tensor.contiguous().view(1, -1)
        elif isinstance(self.ch_axis, list):
            self.ch_shapes = [tensor.shape[i] for i in self.ch_axis]
            dims = list(range(tensor.dim()))  # self.ch_shapes =
            permute_dims = deepcopy(self.ch_axis)
            for dim in dims:
                if dim not in permute_dims:
                    permute_dims.append(dim)
            x = tensor.permute(permute_dims)
            x = x.reshape(int(np.prod(self.ch_shapes)), -1)  # (#channels, -1)
        else:
            raise NotImplementedError("ch axis must be int or list.")
        hist_manager.collect(data=x)

    def _cal_min_max_(self):
        min_val, max_val = self.hist_manager.find_lowest_kl_bound(
            self.dtype.bitwidth, iter_times=self.iter_times
        )
        min_val = min_val.reshape(self.ch_shapes)
        max_val = max_val.reshape(self.ch_shapes)
        self.min_val.resize_(min_val.shape).copy_(min_val)
        self.max_val.resize_(max_val.shape).copy_(max_val)
        return self.min_val, self.max_val


@ObserverRegister.add("mse", "MSE", "Mse")
class MSEObserver(ObserverABC):
    def __init__(
        self,
        granularity="tensor",
        p: float = 2.0,
        hist_bin_num: int = 2048,
        iter_times: int = 1024,
    ):
        super().__init__(granularity)
        self.hist_manager = HistManager(num_bins=hist_bin_num)
        self.ch_shapes = 1
        self.p = p
        self.iter_times = iter_times

    def _update_(self, tensor: torch.Tensor):
        hist_manager = self.hist_manager
        if self.ch_axis == -1:
            x = tensor.view(1, -1)
        elif isinstance(self.ch_axis, list):
            self.ch_shapes = [tensor.shape[i] for i in self.ch_axis]
            dims = list(range(tensor.dim()))  # self.ch_shapes =
            permute_dims = deepcopy(self.ch_axis)
            for dim in dims:
                if dim not in permute_dims:
                    permute_dims.append(dim)
            x = tensor.permute(permute_dims)
            x = x.reshape(int(np.prod(self.ch_shapes)), -1)  # (#channels, -1)
        else:
            raise NotImplementedError("ch axis must be int or list.")
        hist_manager.collect(data=x)

    def lp_loss(self, pred, tgt, weight, dim=None, use_log_weight=False):
        """loss function measured in L_p Norm

        Args:
            pred (torch.Tensor): the quantized tensor.
            tgt (torch.Tensor): the FP original tensor.
            weight (torch.Tensor): the histogram.
            dim (int, optional): the dim for calculating mean. Defaults to None.
            use_log_weight (bool, optional): apply log to the hist as a weight vector for loss.

        Returns:
            torch.Tensor: Loss value for L_p Norm.
        """
        if use_log_weight:
            weight = weight.log2().clip(0)
        return (
            (pred - tgt).abs().pow(self.p).sum(dim)
            if dim
            else (pred - tgt).abs().pow(self.p) * weight
        ).sum(dim=-1)

    @torch.no_grad()
    def tmporary_calculate_scale_zero_point(self, min_val, max_val):
        assert min_val is not None and max_val is not None
        quant_min, quant_max = self.dtype.qmin, self.dtype.qmax
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int32, device=device)

        if self.symmetric:
            scale = torch.max(
                torch.abs(min_val_neg / self.dtype.qmin),
                torch.abs(max_val_pos / self.dtype.qmax),
            )
            scale = torch.max(scale, eps.to(scale.device))
        else:
            scale = ((max_val_pos - min_val_neg) / float(quant_max - quant_min)).abs()
            scale = torch.max(scale, eps.to(scale.device))
            zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    def find_lowest_mse_bound(self):  # TODO 不确定对不对
        """搜索最优的min_bound, max_bound以追求量化前后最小的MSE
        !注意: 目前的实现方法是从左右两边以等长的stride往0点去推, 还没有非对称的实现方法.
        """
        centers = (
            self.hist_manager.bin_edges_mat[..., 1:]
            + self.hist_manager.bin_edges_mat[..., :-1]
        ) / 2  # 算出每个bin的中线.
        new_min = centers[..., 0]
        new_max = centers[..., -1]
        max_centers = torch.max(
            centers[..., 0].abs(), centers[..., -1].abs()
        )  # 每个通道最大的centers, 作为搜索的起点, 向0迭代. shape=(a, b) or (a)
        stride = max(int(self.hist_manager.num_bins / self.iter_times), 1)
        clip_step_num = (
            torch.true_divide(
                input=max_centers.reshape(-1,1), other=(self.hist_manager.bin_width * stride)
            )
            .floor()
            .long()
        )  # shape=(a, b), 每一个元素是s
        clip_step_max = (
            clip_step_num.min()
        )  # 理论上应该是所有元素相等, 但是除法有误差, 取最小的clip次数. 误差应该不大, 顶多差一个clip step.
        # assert torch.all(input=clip_step_num.eq(clip_step_max))
        clip_step_idx_tensor = torch.arange(
            start=0, end=clip_step_max, step=stride, device=new_min.device
        )  # shape=(s) 因为bin_width每个通道都不同, 但是0点是对齐的, 所以 clip_values的各通道长度应该一致.
        clip_step_shape = list(self.hist_manager.hists_mat.shape)
        clip_step_shape[-1] = 1
        clip_step_idx_tensor = clip_step_idx_tensor.repeat(
            repeats=clip_step_shape
        )  # shape=(a, b, s) or (a, s) or (s)
        clip_values = max_centers.unsqueeze(dim=-1) - torch.mul(
            input=clip_step_idx_tensor, other=self.hist_manager.bin_width
        )  # shape=(a, b, s) or (a, s) or (s)
        best_loss = torch.ones_like(input=max_centers) * float(
            "inf"
        )  # shape=(a, b) or (a).
        best_loss = (
            torch.reshape(input=best_loss, shape=self.ch_shapes + [-1]).squeeze()
            if self._ch_axis != -1
            else best_loss
        )
        min_bound_mat = torch.max(
            -clip_values, new_min.unsqueeze(dim=-1)
        )  # shape=(a, b, s) or (a, s). 其中 s 是clip的step数.
        max_bound_mat = torch.min(clip_values, new_max.unsqueeze(dim=-1))
        # 两个边界值都初始化为0, 用于记录搜索的最优结果.
        best_max_clip_mat = torch.ones_like(input=max_bound_mat) * float(
            "inf"
        )  # shape=(a, b, s) or (a, s)
        best_min_clip_mat = torch.ones_like(input=min_bound_mat) * float("-inf")
        # 用centers代表bins的分布, 做量化并对量化前后做MSE.
        # 临时变量, 暂存每一次迭代中当前的min和max
        for step_idx in range(clip_values.shape[-1]):
            min_bound = min_bound_mat[..., step_idx]  # shape=(a, b) or (a)
            max_bound = max_bound_mat[..., step_idx]
            if self._ch_axis != -1:
                min_bound = torch.reshape(input=min_bound, shape=self.ch_shapes)
                max_bound = torch.reshape(input=max_bound, shape=self.ch_shapes)
                centers = torch.reshape(input=centers, shape=self.ch_shapes + [-1])
                best_max_clip_mat = torch.reshape(
                    input=best_max_clip_mat, shape=self.ch_shapes + [-1]
                )
                best_min_clip_mat = torch.reshape(
                    input=best_min_clip_mat, shape=self.ch_shapes + [-1]
                )
            scale, zero_point = self.tmporary_calculate_scale_zero_point(
                min_val=min_bound, max_val=max_bound
            )
            grans = ""
            if isinstance(self.granularity, list):
                for dim_idx in range(len(self.granularity)):
                    grans += "dim{}".format(dim_idx)
                    grans += ","
                grans = grans[:-1]
            elif self.granularity == "tensor":
                grans = self.granularity
            else:
                grans = "dim0"
            quant_param = QuantParam(
                dtype=f"int{self.dtype.bitwidth}",
                scale=scale,
                zero_point=zero_point,
                granularity=grans,
            )
            quant_centers = quant_param.quant_tensor(tensor=centers, simulate=True)
            tmp_loss = self.lp_loss(
                pred=quant_centers,
                tgt=centers,
                weight=torch.reshape(
                    input=self.hist_manager.hists_mat, shape=self.ch_shapes + [-1]
                )
                if self._ch_axis != -1
                else self.hist_manager.hists_mat,
            )
            loss_mask = tmp_loss < best_loss
            best_loss[loss_mask] = tmp_loss[loss_mask]
            best_min_clip_mat[..., step_idx][loss_mask] = min_bound[loss_mask]
            best_max_clip_mat[..., step_idx][loss_mask] = max_bound[loss_mask]
        return (
            best_min_clip_mat.max(dim=-1).values,
            best_max_clip_mat.min(dim=-1).values,
        )

    def _cal_min_max_(self):
        min_val, max_val = self.find_lowest_mse_bound()
        min_val = min_val.reshape(self.ch_shapes)
        max_val = max_val.reshape(self.ch_shapes)
        if self.ch_axis != -1:
            self.min_val.resize_(min_val.shape).copy_(min_val)
            self.max_val.resize_(max_val.shape).copy_(max_val)
        else:
            self.min_val = min_val
            self.max_val = max_val
        return self.min_val, self.max_val


if __name__ == "__main__":
    # from hmquant.ptq.nn_layers.observers.EMAMinMaxObserver import EMAMinMaxObserver
    from .hist_observers import MSEObserver
    from .observer_abc import MinMaxObserver
    from .utils import (
        gen_rand_feat_dataloader,
        simulated_observer_calibration,
        simulated_observer_test,
        print_diff_info,
    )

    out_diff_dict = []
    device = "cuda:3"
    dataloader = gen_rand_feat_dataloader()

    observer_dict = {
        "symmetric": True,
    }
    minmax_observer_pt = MinMaxObserver(**observer_dict)
    simulated_observer_calibration(
        dataloader=dataloader, observer=minmax_observer_pt, device=device
    )
    simulated_observer_test(
        diff_list=out_diff_dict,
        dataloader=dataloader,
        observer=minmax_observer_pt,
        device=device,
    )
    mse_observer_pt = MSEObserver()
    simulated_observer_calibration(
        dataloader=dataloader, observer=mse_observer_pt, device=device
    )
    simulated_observer_test(
        diff_list=out_diff_dict,
        dataloader=dataloader,
        observer=mse_observer_pt,
        device=device,
    )

    observer_dict["granularity"] = "dim1"
    minmax_observer_pc = MinMaxObserver(**observer_dict)
    simulated_observer_calibration(
        dataloader=dataloader, observer=minmax_observer_pc, device=device
    )
    simulated_observer_test(
        diff_list=out_diff_dict,
        dataloader=dataloader,
        observer=minmax_observer_pc,
        device=device,
    )
    mse_observer_pc = MSEObserver(**observer_dict)
    simulated_observer_calibration(
        dataloader=dataloader, observer=mse_observer_pc, device=device
    )
    simulated_observer_test(
        diff_list=out_diff_dict,
        dataloader=dataloader,
        observer=mse_observer_pc,
        device=device,
    )

    observer_dict["granularity"] = ["dim0", "dim1"]
    minmax_observer_pc = MinMaxObserver(**observer_dict)
    simulated_observer_calibration(
        dataloader=dataloader, observer=minmax_observer_pc, device=device
    )
    simulated_observer_test(
        diff_list=out_diff_dict,
        dataloader=dataloader,
        observer=minmax_observer_pc,
        device=device,
    )
    mse_observer_pc = MSEObserver(**observer_dict)
    simulated_observer_calibration(
        dataloader=dataloader, observer=mse_observer_pc, device=device
    )
    simulated_observer_test(
        diff_list=out_diff_dict,
        dataloader=dataloader,
        observer=mse_observer_pc,
        device=device,
    )

    print_diff_info(out_diff_dict)
