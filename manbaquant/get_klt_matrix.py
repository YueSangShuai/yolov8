import torch
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
import argparse
import torch.nn as nn

import functools
from tqdm import tqdm
torch.set_grad_enabled(False)
# import pdb


import itertools
from itertools import permutations

def gram_schmidt(K):
    # 假设 K 是 n x n 矩阵
    n = K.size(1)
    Q = torch.zeros_like(K)
    for i in range(n):
        # 取出 K 的第 i 列
        q = K[:, i]
        # 对当前列向量 q 进行正交化
        for j in range(i):
            q -= torch.dot(Q[:, j], K[:, i]) * Q[:, j]
        # 归一化处理
        Q[:, i] = q / q.norm()
    return Q


'''采用模拟退火算法来进行优化:同时实现值范围和方差的最小化'''
import numpy as np

def compute_combined_loss(W, K, alpha=0.5):
    """
    计算组合损失函数，包括值范围和方差的加权和。
    
    参数:
    - W: 权重矩阵
    - K: KLT 矩阵
    - alpha: 加权系数，控制值范围和方差的相对重要性

    返回:
    - loss: 组合损失值
    """
    result = W @ K
    value_range = result.max() - result.min()
    variance = result.var()

    # 组合损失函数
    loss = alpha * value_range + (1 - alpha) * variance
    return loss

def optimize_klt_matrix_1(W, K, num_iterations=100, alpha=0.5, temperature=1.0, cooling_rate=0.99):
    """
    使用模拟退火优化 KLT 矩阵，以最小化组合目标函数。

    参数:
    - W: 权重矩阵
    - K: 初始 KLT 矩阵
    - num_iterations: 优化的迭代次数
    - alpha: 加权系数
    - temperature: 初始温度（模拟退火）
    - cooling_rate: 温度的下降速率

    返回:
    - best_K: 优化后的 KLT 矩阵
    """
    n_cols = K.size(1)
    best_K = K
    current_loss = compute_combined_loss(W, K, alpha)
    
    for i in range(num_iterations):
        # 生成一个随机的列交换
        col1, col2 = np.random.choice(n_cols, 2, replace=False)
        new_K = best_K.clone()
        new_K[:, [col1, col2]] = new_K[:, [col2, col1]]
        
        # 计算新的损失
        new_loss = compute_combined_loss(W, new_K, alpha)
        
        # 模拟退火条件接受
        if new_loss < current_loss or np.random.rand() < np.exp((current_loss - new_loss).cpu() / temperature):
            best_K = new_K
            current_loss = new_loss
        
        # 降低温度
        temperature *= cooling_rate
    
    return best_K


'''通过列交换的方式优化组合目标函数，可能会破坏正交性。
为了保持正交性，同时实现值范围和方差的最小化，
可以使用一个更为复杂的优化方法，比如投影梯度下降法。'''
def compute_combined_loss(W, K, alpha=0.1):
    """
    计算组合损失函数，包括值范围和方差的加权和。
    
    参数:
    - W: 权重矩阵
    - K: KLT 矩阵
    - alpha: 加权系数，控制值范围和方差的相对重要性

    返回:
    - loss: 组合损失值
    """
    result = W @ K
    value_range = result.max() - result.min()
    variance = result.var()

    # 组合损失函数
    loss = alpha * value_range + (1 - alpha) * variance
    return loss

def project_to_orthogonal(K):
    """
    将矩阵投影到正交矩阵空间，使用 QR 分解方法。

    参数:
    - K: 输入矩阵

    返回:
    - K_orthogonal: 投影后的正交矩阵
    """
    Q, R = torch.linalg.qr(K)
    return Q

def optimize_klt_matrix_2(W, K, num_iterations=100, alpha=0.1, learning_rate=0.01):
    """
    使用投影梯度下降法优化 KLT 矩阵，以最小化组合目标函数。

    参数:
    - W: 权重矩阵
    - K: 初始 KLT 矩阵
    - num_iterations: 优化的迭代次数
    - alpha: 加权系数
    - learning_rate: 梯度下降的学习率

    返回:
    - best_K: 优化后的正交 KLT 矩阵
    """
    X = W
    K = nn.Parameter(K, requires_grad=True)
    
    optimizer = torch.optim.Adam([K], lr=learning_rate)

    for i in range(num_iterations):
        optimizer.zero_grad()

        loss = compute_combined_loss(X, K, alpha)
        loss.backward()

        optimizer.step()

        # 投影到正交矩阵空间
        with torch.no_grad():
            K.copy_(project_to_orthogonal(K))
        if i % 10 == 0:
            print(f'Iteration {i}, Loss: {loss.item()}')

    return K


"""暴力搜索值范围最小loss：优化klt"""
def optimize_klt_matrix_3(K, W):
    # 获取 K 的列数
    n_cols = K.size(1)

    # 计算初始的 W @ K
    best_K = K

    min_loss = compute_combined_loss(W,K)

    # 遍历所有可能的列顺序
    for perm in permutations(range(n_cols)):
        # 对 K 的列进行重排
        K_perm = K[:, perm]
        current_loss = compute_combined_loss(W,K_perm,alpha=0.001)

        # 如果当前的列顺序能够使范围更小，则更新
        if current_loss < min_loss:
            min_loss = current_loss
            best_K = K_perm

    return best_K

def get_act_klt(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_klt = {}

    def stat_tensor(name, tensor):
        shape = tensor.shape
        cov_matrix = torch.cov(tensor.reshape(-1,shape[-1]).double().T)
    
        # 计算协方差矩阵的特征值和特征向量
        # eig_values, klt_matrix = torch.linalg.eig(cov_matrix)
        eig_values, K = torch.eig(cov_matrix.double(), eigenvectors=True)
        if (K @ K.T)[0,0] >0.99 and (K @ K.T)[0,1]<0.0001 :
            print(f"{name} input is orthogonal")
        else:
            # K, S, Vt = torch.linalg.svd(K)
            K = gram_schmidt(K)
            print(f"{name} input is not orthogonal")
        # K = optimize_klt_matrix_1(cov_matrix.float(),K.T.float()).T
        act_klt[name] = K.float()

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    def stat_input_hook_2(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0].permute(0,1,3,2)
        stat_tensor(name, x)

    hooks = []
    from model_vim_quant.vim.normalized_modules import MatMul
    from quantize.int_matmul import QuantMatMul
    from quantize.int_linear import QuantLinear
    for name, m in model.named_modules():
        if "head" in name:continue
        if isinstance(m, (nn.Linear,QuantLinear)):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name)))
        if isinstance(m, (MatMul,QuantMatMul)):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook_2, name=name)))

    # subset_dataloader = itertools.islice(dataloader, num_samples)
    # for batch in tqdm(subset_dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
    #     if isinstance(batch, list):
    #         images, target = batch
    #     else:
    #         images, target = batch["image"], batch["label"]
    #     model(images.to(device))
    for batch in tqdm(dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
        if isinstance(batch, list):
            images, target = batch
        else:
            images, target = batch["image"], batch["label"]
        model(images.to(device))
        break
        

    for h in hooks:
        h.remove()

    return act_klt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2', help='model name')
    parser.add_argument("--resume", type=str, default='saved_checkpoint/vim_t_midclstok_76p1acc.pth')
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument('--scales-output-path', type=str, default='./act_scales/',help='where to save the act scales')
    parser.add_argument('--shifts-output-path', type=str, default='./act_shifts/',help='where to save the act shifts')
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",choices=["wikitext2", "ptb", "c4", "mix","pile"],help="Where to extract calibration data from.",)
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    args = parser.parse_args()
    return args

def vim_generate_act_klt(lm, data_loader_val,args):
    from timm.models import create_model
    import model_vim_quant.vim.models_mamba
    from model_vim_quant.vim.datasets import build_dataset
    
    net_name = args.model
    output_path = "model_vim_quant/saved_checkpoint"
    batch_size = 1
    num_samples = 128
    
    


    args.data_set = 'IMNET'
    args.data_path = "/data01/datasets/imagenet"
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val=torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(batch_size),
        num_workers=batch_size*2,
        pin_memory=True,
        drop_last=False
    )
    
    act_scales = get_act_klt(lm, data_loader_val,num_samples)
    # save_path = os.path.join(output_path,f'{net_name}_act_klt.pt')
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # torch.save(act_scales, save_path)
    return act_scales

def get_llm_act_klt(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_klt = {}

    def stat_tensor(name, tensor):
        shape = tensor.shape
        cov_matrix = torch.cov(tensor.reshape(-1,shape[-1]).double().T)
    
        # 计算协方差矩阵的特征值和特征向量
        eig_values, K = torch.linalg.eigh(cov_matrix)
        # eig_values, K = torch.eig(cov_matrix.double(), eigenvectors=True)
        if (K @ K.T)[0,0] >0.99 and (K @ K.T)[0,1]<0.0001 :
            print(f"{name} input is orthogonal")
        else:
            # K, S, Vt = torch.linalg.svd(K)
            K = gram_schmidt(K)
            print(f"{name} input is not orthogonal")
        # K = optimize_klt_matrix_1(cov_matrix.float(),K.T.float()).T
        act_klt[name] = K.float()

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    def stat_input_hook_2(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0].permute(0,1,3,2)
        stat_tensor(name, x)

    hooks = []
    from quantize.normalized_modules import MatMul
    from quantize.int_matmul import QuantMatMul
    from quantize.int_linear import QuantLinear
    for name, m in model.named_modules():
        if "head" in name:continue
        if isinstance(m, (nn.Linear,QuantLinear)):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name)))
        if isinstance(m, (MatMul,QuantMatMul)):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook_2, name=name)))

    # subset_dataloader = itertools.islice(dataloader, num_samples)
    # for batch in tqdm(subset_dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
    #     if isinstance(batch, list):
    #         images, target = batch
    #     else:
    #         images, target = batch["image"], batch["label"]
    #     model(images.to(device))
    for batch in tqdm(dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
        input = batch[0]
        model(input.to(device))
        break
        

    for h in hooks:
        h.remove()

    return act_klt


def get_llm_weight_klt(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    weight_klt = {}

    def stat_tensor(name, tensor):
        shape = tensor.shape
        cov_matrix = torch.cov(tensor.reshape(-1,shape[-1]).double().T)
    
        # 计算协方差矩阵的特征值和特征向量
        eig_values, K = torch.linalg.eigh(cov_matrix)
        # eig_values, K = torch.eig(cov_matrix.double(), eigenvectors=True)
        if (K @ K.T)[0,0] >0.99 and (K @ K.T)[0,1]<0.0001 :
            print(f"{name} input is orthogonal")
        else:
            # K, S, Vt = torch.linalg.svd(K)
            K = gram_schmidt(K)
            print(f"{name} input is not orthogonal")
        # K = optimize_klt_matrix_1(cov_matrix.float(),K.T.float()).T
        weight_klt[name] = K.T.float()

    def stat_input_hook(m, x, y, name):
        stat_tensor(name, m.weight)

    hooks = []
    from quantize.normalized_modules import MatMul
    from quantize.int_matmul import QuantMatMul
    from quantize.int_linear import QuantLinear
    for name, m in model.named_modules():
        if "head" in name:continue
        if isinstance(m, (nn.Linear,QuantLinear)):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name)))
            
    for batch in tqdm(dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
        input = batch[0]
        model(input.to(device))
        break
        

    for h in hooks:
        h.remove()

    return weight_klt


if __name__ == '__main__':
    torch.cuda.set_device("cuda:2")
    vim_generate_act_klt()