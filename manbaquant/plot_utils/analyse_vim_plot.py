# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
from re import A
from venv import logger
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))


from vim.datasets import build_dataset
from vim.engine import train_one_epoch, evaluate
from vim.losses import DistillationLoss
from vim.samplers import RASampler
from vim.augment import new_data_aug_generator

from contextlib import suppress

import vim.models_mamba

from vim import utils

# log about
import mlflow
import pickle
from utils.fake_quant_utils.function import QuantizedMatMul
from utils.plot_utils.utils import plot_box_data_perchannel_fig, plot_bar_fig, plot_bar3d_fig
from utils.config.attrdict import AttrDict


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # amp about
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=True)

    # if continue with inf
    parser.add_argument('--if_continue_inf', action='store_true')
    parser.add_argument('--no_continue_inf', action='store_false', dest='if_continue_inf')
    parser.set_defaults(if_continue_inf=False)

    # if use nan to num
    parser.add_argument('--if_nan2num', action='store_true')
    parser.add_argument('--no_nan2num', action='store_false', dest='if_nan2num')
    parser.set_defaults(if_nan2num=False)

    # if use random token position
    parser.add_argument('--if_random_cls_token_position', action='store_true')
    parser.add_argument('--no_random_cls_token_position', action='store_false', dest='if_random_cls_token_position')
    parser.set_defaults(if_random_cls_token_position=False)    

    # if use random token rank
    parser.add_argument('--if_random_token_rank', action='store_true')
    parser.add_argument('--no_random_token_rank', action='store_false', dest='if_random_token_rank')
    parser.set_defaults(if_random_token_rank=False)

    parser.add_argument('--local-rank', default=0, type=int)

    parser.add_argument('--use_vim_torch', default=False, type=bool)


    return parser


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # log about
    run_name = args.output_dir.split("/")[-1]
    if args.local_rank == 0:
        mlflow.start_run(run_name=run_name)
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )
      
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

        
    if args.use_vim_torch:
        from vim.vim_torch import MambaConfig,MambaBlock,RMSNorm
        for mamba in model.layers:
            config = MambaConfig(d_model=mamba.mixer.d_model, n_layers=1)
            mamba_torch = MambaBlock(config).to(device)
            mamba_torch.in_proj = mamba.mixer.in_proj
            mamba_torch.conv1d = mamba.mixer.conv1d
            mamba_torch.x_proj = mamba.mixer.x_proj
            mamba_torch.dt_proj = mamba.mixer.dt_proj
            mamba_torch.conv1d_b = mamba.mixer.conv1d_b
            mamba_torch.x_proj_b = mamba.mixer.x_proj_b
            mamba_torch.dt_proj_b = mamba.mixer.dt_proj_b
            mamba_torch.out_proj = mamba.mixer.out_proj
            mamba_torch.A_log = mamba.mixer.A_log
            mamba_torch.A_log_b = mamba.mixer.A_b_log
            mamba_torch.D = mamba.mixer.D
            mamba_torch.D_b = mamba.mixer.D_b
            mamba.mixer = mamba_torch

            rmsnorm = RMSNorm(mamba.norm.weight.shape[0], mamba.norm.eps)
            rmsnorm.weight = mamba.norm.weight
            mamba.norm = rmsnorm

        rmsnorm = RMSNorm(model.norm_f.weight.shape[0], model.norm_f.eps)
        rmsnorm.weight = model.norm_f.weight
        model.norm_f = rmsnorm

    # switch to evaluation mode
    model.eval()
    


    
    if cfg.quantize:
        from utils.fake_quant_utils.fake_quant import quantize_vim_torch
        quantize_vim_torch(model,config=cfg)
        
    register_hooks(model)

    global input_name
    input_name = "fig100"

    # if os.path.exists(f"data/analyse_fig/{input_name}/{quant_name}/"):
    #     from utils.plot_utils.utils import find_images, concat_images
    #     suffixes = count_suffixes(f"data/analyse_fig/{input_name}/{quant_name}/")
    #     for ss in suffixes.keys():
    #         image_paths = find_images(f"data/analyse_fig/{input_name}/{quant_name}/", "", ss)
    #         filter_images = [image for image in image_paths if image.split(".")[1].isdigit()]
    #         sorted_images = sorted(filter_images,key=lambda x: int(x.split(".")[1]))
    #         concat_images(sorted_images, 6, f"data/analyse_fig/{input_name}/{quant_name}_cat/{ss}")

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for images, target in metric_logger.log_every(data_loader_val, 10, header):
        images = images.to(device, non_blocking=True)
        model(images)
        break
    
    if os.path.exists(f"data/analyse_fig/{input_name}/{quant_name}/"):
        from utils.plot_utils.utils import find_images, concat_images
        suffixes = count_suffixes(f"data/analyse_fig/{input_name}/{quant_name}/")
        for ss in suffixes.keys():
            image_paths = find_images(f"data/analyse_fig/{input_name}/{quant_name}/", "", ss)
            filter_images = [image for image in image_paths if image.split(".")[1].isdigit()]
            sorted_images = sorted(filter_images,key=lambda x: int(x.split(".")[1]))
            os.makedirs(f"data/analyse_fig/{input_name}/{quant_name}_cat/", exist_ok=True)
            concat_images(sorted_images, 6, f"data/analyse_fig/{input_name}/{quant_name}_cat/{ss}")

q_cfg = AttrDict(dict(dim='', # '':pertensor
                      observer={'method':'minmax',
                                "percentile":"0.999999",},
                      n_bit=8,
                    ))
cfg=AttrDict(dict(quantize=False,
                    w_cfg=q_cfg,
                    i_cfg=q_cfg,
                    o_cfg="",))

def count_suffixes(directory):
    """
    统计指定文件夹中图片文件名中'mixer.'后面部分的后缀名称及其数量。
    
    参数:
    - directory: 包含图片的文件夹路径。
    
    返回:
    - 一个字典，键为后缀名称，值为该后缀名称出现的次数。
    """
    suffix_counts = {}
    for filename in os.listdir(directory):
        # 分割文件名以找到'mixer.'后面的部分
        parts = filename.split('mixer.')
        if len(parts) > 1:
            suffix = parts[-1]  # 获取'mixer.'后面的部分
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return suffix_counts

from utils.fake_quant_utils.fake_quant import QLinear, QConv1d, QMatMul
def analyse_hook(module, input, output):  # 新增函数处理层的输出
    module_name = module_to_name.get(module, "Unnamed module")
    os.makedirs(f"data/analyse_fig/{input_name}/{quant_name}/", exist_ok=True)
    save_dir = f"data/analyse_fig/{input_name}/{quant_name}"
    
    # # 分析权重（在前向传播前执行，但这里为了简洁放在一起展示）
    # if isinstance(module, (QLinear, QConv1d)):
    #     weight = module.weight.data
    #     plot_box_data_perchannel_fig(weight, f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg", axis=-1)
    #     plot_bar_fig(weight, f"{save_dir}/{module_name}_weight_bar_data.jpg")
    #     plot_bar3d_fig(weight, f"{save_dir}/{module_name}_weight_bar3d_data.jpg")
    
    for i,temp_input in enumerate(input):
        if isinstance(temp_input, torch.Tensor):  # 确保有输入且为Tensor类型
            plot_box_data_perchannel_fig(temp_input[0], f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg", axis=-1)
            plot_bar_fig(temp_input[0], f"{save_dir}/{module_name}_input{i}_bar_data.jpg")
            plot_bar3d_fig(temp_input[0], f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")
    
    # 分析输出
    if isinstance(output, torch.Tensor):  # 或者 isinstance(outputs, tuple) if模块有多个输出
        plot_box_data_perchannel_fig(output[0], f"{save_dir}/{module_name}_output_box_data_perchannel.jpg", axis=-1)
        plot_bar_fig(output[0], f"{save_dir}/{module_name}_output_bar_data.jpg")
        plot_bar3d_fig(output[0], f"{save_dir}/{module_name}_output_bar3d_data.jpg")

def analyse_hook_2(module, input, output):  # 新增函数处理层的输出
    module_name = module_to_name.get(module, "Unnamed module")
    os.makedirs(f"data/analyse_fig/{input_name}/{quant_name}/", exist_ok=True)
    save_dir = f"data/analyse_fig/{input_name}/{quant_name}"
    
    for i,temp_input in enumerate(input):
        if isinstance(temp_input, torch.Tensor):  # 确保有输入且为Tensor类型
            plot_box_data_perchannel_fig(torch.amax(temp_input[0],dim=0), f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg", axis=-1)
            plot_bar_fig(torch.amax(temp_input[0],dim=0), f"{save_dir}/{module_name}_input{i}_bar_data.jpg")
            plot_bar3d_fig(torch.amax(temp_input[0],dim=0), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")
    
    # 分析输出
    if isinstance(output, torch.Tensor):  # 或者 isinstance(outputs, tuple) if模块有多个输出
        plot_box_data_perchannel_fig(torch.amax(output[0],dim=0), f"{save_dir}/{module_name}_output_box_data_perchannel.jpg", axis=-1)
        plot_bar_fig(torch.amax(output[0],dim=0), f"{save_dir}/{module_name}_output_bar_data.jpg")
        plot_bar3d_fig(torch.amax(output[0],dim=0), f"{save_dir}/{module_name}_output_bar3d_data.jpg")

def register_hooks(model):
    global module_to_name 
    global quant_name
    quant_name = "fp_data" if not cfg.quantize else "w8a8_data"
    module_to_name = {module: name for name, module in model.named_modules()}
    for i,layer in enumerate(model.layers):

        # 新增后处理hook
        # layer.mixer.conv1d.register_forward_hook(analyse_hook)
        # layer.mixer.conv1d_b.register_forward_hook(analyse_hook)
        # layer.mixer.in_proj.register_forward_hook(analyse_hook)
        layer.mixer.x_proj.register_forward_hook(analyse_hook)
        layer.mixer.x_proj_b.register_forward_hook(analyse_hook)
        # layer.mixer.dt_proj.register_forward_hook(analyse_hook)
        # layer.mixer.dt_proj_b.register_forward_hook(analyse_hook)
        # layer.mixer.out_proj.register_forward_hook(analyse_hook)
        # layer.mixer.matmul.register_forward_hook(analyse_hook_2)
        # layer.mixer.matmul_b.register_forward_hook(analyse_hook_2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
