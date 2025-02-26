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


def get_act_scales(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    def stat_input_hook_2(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0].squeeze(2).permute(0,2,1)
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))
        if isinstance(m, (nn.Conv1d,nn.Conv2d)):
            if 'patch_embed' in name:continue
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook_2, name=name)))

    subset_dataloader = itertools.islice(dataloader, num_samples)
    for batch in tqdm(subset_dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
        if isinstance(batch, list):
            images, target = batch
        else:
            images, target = batch["image"], batch["label"]
        model(images.to(device))

    for h in hooks:
        h.remove()

    return act_scales

def get_act_shifts(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_shifts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 *((comming_max+comming_min)/2)
        else:
            act_shifts[name] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    subset_dataloader = itertools.islice(dataloader, num_samples)
    for batch in tqdm(subset_dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
        images,targets = batch
        model(images.to(device))

    for h in hooks:
        h.remove()

    return act_shifts


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


@torch.no_grad()
def vim_generate_act_scale_shift(args):
    from timm.models import create_model
    import model_vim_quant.vim.models_mamba
    from model_vim_quant.vim.datasets import build_dataset
    
    # args = parse_args()
    
    resum_path = args.resume
    model_name = args.model
    net_name = args.model.split("_")[0]+"-"+args.model.split("_")[1]
    # resum_path = "model_vim_quant/saved_checkpoint/vim_b_midclstok_81p9acc.pth"
    # model_name = "vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2"
    # net_name = "vim-base"
    # resum_path = "vim_quant/saved_checkpoint/vim_s_midclstok_80p5acc.pth"
    # model_name = "vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
    # net_name = "vim-small"
    # resum_path = "model_vim_quant/saved_checkpoint/vim_t_midclstok_76p1acc.pth"
    # model_name = "vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
    # net_name = "vim-tiny"
    output_path = "./saved_checkpoint"
    batch_size = 1
    num_samples = 128
    
    
    device = torch.device('cuda')

    lm  = create_model(
        model_name,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224
    )
    lm.to(device)
    lm.eval()

    checkpoint = torch.load(resum_path, map_location='cpu')
    lm.load_state_dict(checkpoint['model'])
    from model_vim_quant.vim.utils import convert_vim_2_vim_torch
    convert_vim_2_vim_torch(lm,device)

    args.data_set = 'IMNET'
    args.data_path = "/data01/datasets/imagenet"
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val=torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(batch_size),
        num_workers=32,
        pin_memory=True,
        drop_last=False
    )
    
    act_scales = get_act_scales(lm, data_loader_val,num_samples)
    save_path = os.path.join(output_path,f'{net_name}_scale.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)
    print(f"savee to {save_path}")

    act_shifts = get_act_shifts(lm, data_loader_val,num_samples)
    save_path = os.path.join(output_path,f'{net_name}_shift.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_shifts, save_path)
    print(f"savee to {save_path}")
    return act_scales,act_shifts


def mamba2d_classify_generate_act_scale_shift():
    from mmengine.config import Config, ConfigDict, DictAction
    from mmengine.runner import Runner
    from model_image_classification.src.mamba import Mamba2DModel
    from model_image_classification.utils.datasets import build_dataset
    args = parse_args()
    model_cfg = './model_image_classification/config/mamba2d.py'
    cfg = Config.fromfile(model_cfg)
    cfg.model_ckpt= "./ckpt/mamba2d_s.pth"
    cfg.work_dir = './work_dirs/mamba2d'
    runner = Runner.from_cfg(cfg)
    
    runner._test_loop = runner.build_test_loop(runner._test_loop)  # type: ignore

    runner.call_hook('before_run')

    # make sure checkpoint-related hooks are triggered after `before_run`
    runner.load_or_resume()
    runner.hooks[1]._swap_ema_parameters()
    
    from model_image_classification.utils.utils import convert_vim_2_vim_torch
    convert_vim_2_vim_torch(runner.model.backbone,"cuda")
    output_path = "image_classification/ckpt/"
    net_name = cfg.model_ckpt.split("/")[-1].split(".")[0]
    batch_size = 1
    num_samples = 128
    
    
    device = torch.device('cuda')

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
    
    act_scales = get_act_scales(runner.model, data_loader_val,num_samples)
    save_path = os.path.join(output_path,f'{net_name}_scale.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)

    act_shifts = get_act_shifts(runner.model, data_loader_val,num_samples)
    save_path = os.path.join(output_path,f'{net_name}_shift.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_shifts, save_path)


def mamba3d_video_generate_act_scale_shift(runner):
    output_path = "ckpt/"
    net_name = "ucf101"

    
    act_scales = {}
    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in runner.model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))
    # metrics = runner.test_loop.run()
    for h in hooks:
        h.remove()
    # save_path = os.path.join(output_path,f'{net_name}_scale.pt')
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # torch.save(act_scales, save_path)

    act_shifts = {}
    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 *((comming_max+comming_min)/2)
        else:
            act_shifts[name] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in runner.model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )
    metrics = runner.test_loop.run()
    for h in hooks:
        h.remove()
    save_path = os.path.join(output_path,f'{net_name}_shift.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_shifts, save_path)

def mamband_seg_generate_act_scale_shift(model,dataloader,model_name):

    
    output_path = "./ckpt/"
    net_name = model_name
    batch_size = 1
    num_samples = 128
    
    
    model.keywords['predictor'].eval()
    device = next(model.keywords['predictor'].parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.keywords['predictor'].named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    subset_dataloader = itertools.islice(dataloader, num_samples)
    for batch in tqdm(subset_dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
        if isinstance(batch, list):
            images, target = batch
        else:
            images, target = batch["image"], batch["label"]
        model(images.to(device))

    for h in hooks:
        h.remove()

    save_path = os.path.join(output_path,f'{net_name}_scale.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)

    act_shifts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 *((comming_max+comming_min)/2)
        else:
            act_shifts[name] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.keywords['predictor'].named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    subset_dataloader = itertools.islice(dataloader, num_samples)
    for batch in tqdm(subset_dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
        if isinstance(batch, list):
            images, target = batch
        else:
            images, target = batch["image"], batch["label"]
        model(images.to(device))

    for h in hooks:
        h.remove()

    save_path = os.path.join(output_path,f'{net_name}_shift.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_shifts, save_path)

def mamba_llm_generate_act_scale(model):
    import torch,datasets,random
    from quantize import QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d
    num_samples = 128
    device = torch.device('cuda')
    traindata = datasets.load_dataset("hellaswag",split='test')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    trainenc = tokenizer("\n\n".join(traindata['ctx']), return_tensors='pt') 
    trainloader = []
    for _ in range(128):
        i = random.randint(0, trainenc.input_ids.shape[1] - 2048 - 1)
        j = i + 2048
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    
    act_scales = {}
    def stat_tensor(name, tensor):
        if "conv1d" in name:
            tensor = tensor.permute(0,2,1)
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear,nn.Conv1d,QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d)):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))  
    
    subset_dataloader = itertools.islice(trainloader, num_samples)

    for batch in tqdm(subset_dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
        input = batch[0]
        model(input.to(device))

    for h in hooks:
        h.remove()

    return act_scales


def mamba_llm_generate_act_shift(model):
    import torch,datasets,random
    from quantize import QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d
    num_samples = 128
    device = torch.device('cuda')
    traindata = datasets.load_dataset("hellaswag",split='test')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    trainenc = tokenizer("\n\n".join(traindata['ctx']), return_tensors='pt') 
    trainloader = []
    for _ in range(128):
        i = random.randint(0, trainenc.input_ids.shape[1] - 2048 - 1)
        j = i + 2048
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        
    subset_dataloader = itertools.islice(trainloader, num_samples)
    
    model.eval()
    device = next(model.parameters()).device
    act_shifts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 *((comming_max+comming_min)/2)
        else:
            act_shifts[name] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear,nn.Conv1d,QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d)):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    for batch in tqdm(subset_dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
        images,targets = batch
        model(images.to(device))

    for h in hooks:
        h.remove()

    return act_shifts

def mamba_llm_generate_channel_mean(model):
    import torch,datasets,random
    from quantize import QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d
    num_samples = 128
    device = torch.device('cuda')
    traindata = datasets.load_dataset("hellaswag",split='test')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    trainenc = tokenizer("\n\n".join(traindata['ctx']), return_tensors='pt') 
    trainloader = []
    for _ in range(128):
        i = random.randint(0, trainenc.input_ids.shape[1] - 2048 - 1)
        j = i + 2048
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        
    subset_dataloader = itertools.islice(trainloader, num_samples)
    
    model.eval()
    device = next(model.parameters()).device
    act_shifts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).detach()
        comming_max = torch.quantile(tensor,0.75,dim=0)
        comming_min = torch.quantile(tensor,0.25,dim=0)
        # comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        # comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.9*act_shifts[name] + 0.1 *((comming_max+comming_min)/2)
        else:
            act_shifts[name] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear,nn.Conv1d,QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d)):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    for batch in tqdm(subset_dataloader,desc="Processing batches", dynamic_ncols=True, leave=True):
        images,targets = batch
        model(images.to(device))

    for h in hooks:
        h.remove()

    return act_shifts

if __name__ == '__main__':
    vim_generate_act_scale_shift()
    # mamba2d_classify_generate_act_scale_shift()
    # mamba3d_video_generate_act_scale_shift()