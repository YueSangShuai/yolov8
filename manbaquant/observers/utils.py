from copy import deepcopy
import os
from typing import List
import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from ._hist_manager import HistManager


try:
    from tqdm import tqdm
except:
    pass


def merge_hists_of_observers(hist_manager_list: List[HistManager]):
    """用于合并来自多个observer的直方图, 并返回一个统一的HistManager对象.

    Args:
        hist_manager_list (List[HistManager]): 来自多个Observer对象的HistManager对象组成的list.
    """
    assert hist_manager_list is not None
    bin_width = hist_manager_list[0].bin_width
    channel_shape = list(hist_manager_list[0].hists_mat.shape[: -1])
    # 统计全局信息: 0点左侧的bin个数最多是多少, 0点右侧的bin最多是多少.
    max_left_bin_num = 0
    max_right_bin_num = 0
    for tmp_hist_manager in hist_manager_list:
        # 首先保证各个tmp_hist_manager的bin_width一致.
        assert (tmp_hist_manager.bin_width == bin_width).all(), "Bin witdh is not consistent."
        # 其次保证各个tmp_hist_manager的channel num一致.
        assert list(tmp_hist_manager.hists_mat.shape[: -1]) == channel_shape, "Data channel shape must be the same."
        # 再统计最大的左侧bin个数, 右侧的bin个数.
        if max_left_bin_num < tmp_hist_manager.left_bins_num.max().item():
            max_left_bin_num = tmp_hist_manager.left_bins_num.max().item()
        if max_right_bin_num < tmp_hist_manager.right_bins_num.max().item():
            max_right_bin_num = tmp_hist_manager.right_bins_num.max().item()
    # 创建了 hist 和 bin_edges.
    hist_manager = HistManager(num_bins=max_left_bin_num + max_right_bin_num - 1)
    hist_channel_shape = deepcopy(channel_shape)
    hist_channel_shape.append(max_left_bin_num + max_right_bin_num - 1)
    hist_manager.hists_mat = torch.zeros(
        size=hist_channel_shape,
        device=hist_manager_list[0].hists_mat.device
    )
    # 构建了bin_edges.
    bin_edges_shape = deepcopy(list(hist_manager.hists_mat.shape))
    bin_edges_shape[-1] += 1
    # 从0左侧的第一个edge往左侧减, 一直要减max_left_bin_num-1步.
    left_repeat_shape = deepcopy(bin_edges_shape)
    left_repeat_shape[-1] = 1
    zero_to_left_step_tensor = torch.arange(
        start=0,
        end=-max_left_bin_num - 1,
        step=-1,
        device=hist_manager.hists_mat.device
    ).repeat(repeats=left_repeat_shape)
    left_step_tensor = torch.mul(
        input=bin_width.unsqueeze(dim=-1),
        other=zero_to_left_step_tensor
    )
    left_bin_edges_mat = torch.add(
        input=bin_width.unsqueeze(dim=-1) / 2.0,
        other=left_step_tensor
    ).flip(dims=(-1,))
    # 从0右侧的第一个edge往右侧加, 一直要减max_right_bin_num-1步.
    right_repeat_shape = deepcopy(bin_edges_shape)
    right_repeat_shape[-1] = 1
    zero_to_right_step_tensor = torch.arange(
        start=0,
        end=max_right_bin_num + 1,
        step=1,
        device=hist_manager.hists_mat.device
    ).repeat(repeats=right_repeat_shape)
    right_step_tensor = torch.mul(
        input=bin_width.unsqueeze(dim=-1),
        other=zero_to_right_step_tensor
    )
    right_bin_edges_mat = torch.add(
        input=-bin_width.unsqueeze(dim=-1) / 2.0,
        other=right_step_tensor
    )
    # 拼接成完整的bin_edges.
    hist_manager.bin_edges_mat = torch.cat(
        tensors=(left_bin_edges_mat[..., :-1], right_bin_edges_mat[..., 1:]),
        dim=-1
    )
    
    for tmp_hist_manager in hist_manager_list:
        # 先检查0点左侧有多少个bin, 这样就知道要向左侧pad多少个frec=0的bin.
        left_pad_bin_num = max_left_bin_num - tmp_hist_manager.left_bins_num.max().item()
        if len(tmp_hist_manager.left_bins_num.unsqueeze(dim=-1).shape) == 1:
            hist_manager.hists_mat[left_pad_bin_num: left_pad_bin_num + len(tmp_hist_manager.hists_mat.shape[-1])] += tmp_hist_manager.hists_mat
        elif len(tmp_hist_manager.left_bins_num.unsqueeze(dim=-1).shape) == 2:
            for row_idx in range(tmp_hist_manager.hists_mat.shape[0]):
                hist_manager.hists_mat[row_idx, left_pad_bin_num: left_pad_bin_num + tmp_hist_manager.hists_mat.shape[-1]] += tmp_hist_manager.hists_mat[row_idx, :]
        elif len(tmp_hist_manager.left_bins_num.unsqueeze(dim=-1).shape) == 3:
            for row_idx in range(tmp_hist_manager.hists_mat.shape[0]):
                for col_idx in range(tmp_hist_manager.hists_mat.shape[1]):
                    hist_manager.hists_mat[row_idx, col_idx, left_pad_bin_num: left_pad_bin_num + tmp_hist_manager.hists_mat.shape[-1]] += tmp_hist_manager.hists_mat[row_idx, col_idx, :]
    return hist_manager
        
        
@torch.no_grad()
def pure_diff(raw_tensor: torch.Tensor, quanted_tensor: torch.Tensor):
    """计算两个tensor之间的平均L1误差, 平均相对L1误差, 相对MSE, 余弦距离.

    Args:
        raw_tensor (torch.Tensor): 伪量化前tensor
        quanted_tensor (torch.Tensor): 伪量化后tensor

    Returns:
        (tensor, tensor, tensor, tensor): 平均L1误差, 平均相对L1误差, 相对MSE, 余弦距离.
    """
    diff_tensor = torch.sub(input=raw_tensor, other=quanted_tensor)
    abs_tensor = torch.abs(diff_tensor)
    mean_ab_tensor = torch.mean(abs_tensor)
    mean_re_tensor = torch.true_divide(torch.mean(abs_tensor), torch.mean(torch.abs(raw_tensor)) + 1e-6)
    related_mse = torch.sum((diff_tensor) ** 2) / (torch.sum(raw_tensor ** 2) + 1e-7)
    cos_distance = 1 - torch.nn.functional.cosine_similarity(raw_tensor.view(-1), quanted_tensor.view(-1), dim=-1)
    del diff_tensor, abs_tensor
    return mean_ab_tensor, mean_re_tensor, related_mse, cos_distance
    

def record_error_into_table(error_table_list: list, observer, op_name: str = "unknow"):
    """用于比较一个Observer对fp_tensor伪量化前后的绝对误差, 相对误差和信噪比, 用于证明Observer参数搜索的正确性.

    Args:
        error_table_list (list): 每一个元素将会记录一个observer在一种granularity下的量化效果.
        observer (ObserverBase): 实例化的observer.
        fp_tensor (torch.Tensor): FP的tensor
        quant_tensor (torch.Tensor): quant_param伪量化后的tensor.
        op_name (str): observer绑定的算子名称. (算子所在nn_layer文件名)
    """
    error_table_list.append({
            "op": op_name,
            "observer": observer.observer_name,
            "channel": observer._ch_axis,
            "managed": "no" if observer.manager is None else "yes",
            "mean L1 Error": observer.error_record_dict["mean L1 Error"] / observer.batch_count,
            "mean related L1 Error": observer.error_record_dict["mean related L1 Error"] / observer.batch_count,
            "related mse": observer.error_record_dict["related mse"] / observer.batch_count,
            "cosine dist": observer.error_record_dict["cosine dist"] / observer.batch_count
        })

    
def print_diff_info(diff_out: list, op_name: str = "unkow", save_path: str = None):
    """使用rich table工具输出Observer对FP量化前后的效果.

    Args:
        diff_out (list): 待输出的表现汇总记录. Default to "unkown".
        filename (str): 输出表格的文本文件目录, 后面回追加{op_name}.txt. 如果是None就只输出控制台. Default to None.
    """
    table = Table(title="diff table")
    table.add_column("op")
    table.add_column("observer")
    table.add_column("channel axis")
    table.add_column("managed")
    table.add_column("abs mean diff")
    table.add_column("abs mean related diff")
    table.add_column("mean related mse")
    table.add_column("mean cosine distance")
    for item in diff_out:
        table.add_row(
            str(item["op"]),
            str(item["observer"]),
            str(item["channel"]), 
            str(item["managed"]), 
            str(item["mean_ab_diff"].cpu().numpy()), 
            str(item["mean_re_diff"].cpu().numpy()), 
            str(item["mean_re_mse"].cpu().numpy()),
            str(item["mean_cos_dist"].cpu().numpy())
        )
    if save_path is not None:
        save_path_str = os.path.join(save_path, op_name) + ".txt"
        with open(save_path_str, "w") as f:
            console = Console(file=[f, None])
    else:
        console = Console()
    console.print(table)
    

def gen_rand_feat_dataloader(batch_size: int = 32, feat_chs: int = 64, feat_height: int = 128, feat_width: int = 128, datasize: int = 128):
    """生成随机的特征图dataloader, 目的是模拟calibration.

    Args:
        batch_size (int, optional): the first dim of feature map. Defaults to 32.
        feat_chs (int, optional): the second dim (input channels) of feature map. Defaults to 64.
        feat_height (int, optional): the third dim (height) of feature map. Defaults to 128.
        feat_width (int, optional): the forth dim (width) of feature map. Defaults to 128.
        datasize (int, optional): the number of batch. Defaults to 128.

    Returns:
        _type_: _description_
    """
    batch_feat_list = []
    for _ in range(datasize):
        tensor = torch.randn(
            size=(batch_size, feat_chs, feat_height, feat_width), 
        )
        batch_feat_list.append((tensor, None))
    return batch_feat_list
        
    
def simulated_observer_calibration(dataloader: DataLoader, observer, device: str):
    """simulte the calibration step using observer.

    Args:
        dataloader (DataLoader): the dataloader we create in random.
        observer (ObserverBase): the observer we used in the calibration.
        device (str): cuda:0.
    """
    for batch_feat, _ in tqdm(dataloader):
        batch_feat = batch_feat.to(device)
        observer.update(batch_feat)

        
def simulated_observer_test(diff_list: list, dataloader: DataLoader, observer, device: str):
    """simulte the test stage for calibrated observer which has already gotten the quant parameters.

    Args:
        diff_list (list): the record list for abs/relative diffs and SNR we will return when testing.
        dataloader (DataLoader): the dataloader we have created
        observer (ObserverBase): the observer we test.
        device (str): cuda:0.
    """
    mean_ab, mean_re, mean_snr, mean_cos = 0.0, 0.0, 0.0, 0.0
    count = 0.0
    quant_param = observer.calculate_qparams()
    for tensor, _ in tqdm(dataloader):
        tensor = tensor.to(device)
        quant_tensor = quant_param.quant_tensor(tensor=tensor)
        mean_ab_tensor, mean_re_tensor, related_mse, cos_distance = pure_diff(raw_tensor=tensor, quanted_tensor=quant_tensor)
        mean_ab += mean_ab_tensor
        mean_re += mean_re_tensor
        mean_snr += related_mse
        mean_cos += cos_distance
        count += 1.0
    mean_ab /= count
    mean_re /= count
    mean_snr /= count
    mean_cos /= count
    diff_list.append({
            "observer": observer.observer_name,
            "channel": observer._ch_axis,
            "mean_ab_diff": mean_ab,
            "mean_re_diff": mean_re,
            "mean_re_mse": mean_snr,
            "mean_cos_dist": mean_cos,
            "managed": "no" if observer.manager is None else "yes",
            "op": "unknow"
        })
   
    
def gen_rand_feat(min, max, batch_size=32, feat_chs=16, feat_height=224, feat_width=224):
    """根据传入参数随机生成一个特征图Tensor.

    Args:
        min (_type_): feature的最小值
        max (_type_): feature的最大值
        batch_size (int, optional): Defaults to 32.
        feat_chs (int, optional): Defaults to 16.
        feat_height (int, optional): Defaults to 224.
        feat_width (int, optional): Defaults to 224.

    Returns:
        torch.Tensor: 随机生成的特征图.
    """
    tensor = torch.randn(
        size=(batch_size, feat_chs, feat_height, feat_width), 
    )
    mapped_tensor = (tensor - tensor.min()) * (max - min) / (tensor.max() - tensor.min()) + min
    return mapped_tensor

def analysis_dim(granularities):
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
            if isinstance(granularity,str):
                assert len(granularity==4) and granularity[:3] == "dim"
                ch_axis.append(int(granularity[3:]))
            elif isinstance(granularity,int):
                ch_axis.append(granularity)
            else:
                raise NotImplemented
        for ch in ch_axis:
            assert ch >= 0, "for stability"
    elif granularities == "tensor":
        ch_axis = -1
    elif granularities[:3] == "dim":
        ch_axis = [
            int(granularities[3:]),
        ]
    return ch_axis

