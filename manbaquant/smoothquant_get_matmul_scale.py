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

import os
import sys
ROOT = os.getcwd()
sys.path.append(str(ROOT)+"/vim_quant")
torch.cuda.set_device(7)

def get_matmul_act_scales_S3(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}
    from quantize.int_matmul import QuantMatMul
    def stat_tensor(name, tensor):
        tmp = tensor.permute(0,1,3,2)
        hidden_dim = tmp.shape[-1]
        tmp = tmp.reshape(-1, hidden_dim).abs().detach()
        # comming_max = torch.max(tmp, dim=0)[0].float().cpu()
        comming_max = torch.quantile(tmp, 0.999999, dim=0).float().cpu()
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
        if isinstance(m, QuantMatMul):
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

    return act_scales
def get_matmul_act_scales_S4(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_scales_x1 = {}
    act_scales_x2 = {}
    act_scales = {}
    from quantize.int_matmul import QuantMatMul
    def stat_tensor(name, tensor):
        x1 = tensor[0]
        hidden_dim = x1.shape[-1]
        x1 = x1.reshape(-1, hidden_dim).abs().detach()
        comming_max = torch.max(x1, dim=0)[0].float().cpu()
        # comming_max = torch.quantile(tmp, 0.999999, dim=0).float().cpu()
        if name in act_scales_x1:
            act_scales_x1[name] = torch.max(act_scales_x1[name], comming_max)
        else:
            act_scales_x1[name] = comming_max

        x2 = tensor[1].permute(0,1,3,2)
        hidden_dim = x2.shape[-1]
        x2 = x2.reshape(-1, hidden_dim).abs().detach()
        comming_max = torch.max(x2, dim=0)[0].float().cpu()
        # comming_max = torch.quantile(tmp, 0.999999, dim=0).float().cpu()
        if name in act_scales_x2:
            act_scales_x2[name] = torch.max(act_scales_x2[name], comming_max)
        else:
            act_scales_x2[name] = comming_max

    def stat_input_hook(m, x, y, name):
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, QuantMatMul):
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

    alpha = 0.5
    for key,val in act_scales_x1.items():
        act_scales[key] = ((val.pow(alpha) / act_scales_x2[key].pow(1 - alpha)).clamp(min=1e-5).to(val))
    return act_scales



def get_matmul_act_scales(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}
    from quantize.int_matmul import QuantMatMul
    def stat_tensor(name, tensor):
        tmp = tensor.permute(0,1,3,2)
        hidden_dim = tmp.shape[-1]
        tmp = tmp.reshape(-1, hidden_dim).abs().detach()
        # comming_max = torch.max(tmp, dim=0)[0].float().cpu()
        comming_max = torch.quantile(tmp, 0.999999, dim=0).float().cpu()
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
        if isinstance(m, QuantMatMul):
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

    return act_scales

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2', help='model name')
    parser.add_argument("--resume", type=str, default='saved_checkpoint/vim_t_midclstok_76p1acc.pth')
    parser.add_argument("--batch_size", type=int, default=10, help="batch size.")
    parser.add_argument('--scales-output-path', type=str, default='./act_scales/',help='where to save the act scales')
    parser.add_argument('--shifts-output-path', type=str, default='./act_shifts/',help='where to save the act shifts')
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",choices=["wikitext2", "ptb", "c4", "mix","pile"],help="Where to extract calibration data from.",)
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    args = parser.parse_args()
    return args


@torch.no_grad()
def vim_generate_matmul_scale():
    from timm.models import create_model
    import model_vim_quant.vim.models_mamba
    from model_vim_quant.vim.datasets import build_dataset
    args = parse_args()
    # torch.cuda.set_device("cuda:6")
    
    # resum_path = "model_vim_quant/saved_checkpoint/vim_t_midclstok_76p1acc.pth"
    # model_name = "vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
    # net_name = "vim-tiny"
    resum_path = "model_vim_quant/saved_checkpoint/vim_t+_midclstok_ft_78p3acc.pth"
    model_name = "vim_tinyplus_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
    net_name = "vim-tinyplus"
    # resum_path = "model_vim_quant/saved_checkpoint/vim_s_midclstok_80p5acc.pth"
    # model_name = "vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
    # net_name = "vim-small"
    # resum_path = "model_vim_quant/saved_checkpoint/vim_s+_midclstok_ft_81p6acc.pth"
    # model_name = "vim_smallplus_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
    # net_name = "vim-smallplus"
    # resum_path = "model_vim_quant/saved_checkpoint/vim_b_midclstok_81p9acc.pth"
    # model_name = "vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2"
    # net_name = "vim-base"
    output_path = "./saved_checkpoint"
    matmul_scale_type = "S4"
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
        num_workers=batch_size*2,
        pin_memory=True,
        drop_last=False
    )
    
    model = lm
    
    #ptq
    from model_vim_quant.vim.normalized_modules import MatMul
    from quantize import QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d
    w_cfg = {"dynamic_method":"per_tensor","n_bits":8}
    a_cfg = {"dynamic_method":"per_tensor","n_bits":8}
    def replace_layers(model, target_class, replacement_class):
        for name, child in model.named_children():
            if isinstance(child, target_class):
                # Replace the layer with the new quantized version
                if target_class == MatMul:
                    setattr(model, name, replacement_class(x1_quant_params=w_cfg,x2_quant_params=a_cfg))
                else:
                    setattr(model, name, replacement_class(child,weight_quant_params=w_cfg,act_quant_params=a_cfg))
            else:
                # Recursively call this function on the child module
                replace_layers(child, target_class, replacement_class)

    # Usage example:
    # Assuming QuantMatMul, QuantLinear, QuantConv1d, QuantConv2d are defined
    replace_layers(model, MatMul, QuantMatMul)
    replace_layers(model, nn.Linear, QuantLinear)
    replace_layers(model, nn.Conv1d, QuantConv1d)
    replace_layers(model, nn.Conv2d, QuantConv2d)
    from quantize.utils import set_quant_state
    set_quant_state(model,weight_quant=True,act_quant=True)

    from quantize.hm_model_utils import fuse_layer_norms, RotateModule, RQuantLinear
    from quantize.hadmard import random_hadamard_matrix
    h1 = random_hadamard_matrix(model.layers[0].mixer.in_proj.in_features,device)
    R1 = RotateModule(h1)
    h2 = random_hadamard_matrix(model.layers[0].mixer.out_proj.in_features,device)
    R2 = RotateModule(h2)
    model.register_parameter("R1",R1.weight)
    
    h3 = random_hadamard_matrix(model.layers[0].mixer.out_proj.in_features,device)
    R3 = RotateModule(h3)

    h4 = random_hadamard_matrix(16,device)
    R4 = RotateModule(h4)
    
    if True:
        
        fuse_layer_norms(model)
        def substitute_layers(model):
            for name,module in model.named_modules():
                if 'in_proj' in name and 'quantizer' not in name:
                    new_module = RQuantLinear(module,R1=R1,transpose=False)
                elif 'out_proj' in name and 'quantizer' not in name:
                    new_module = RQuantLinear(module,R1=R1,R2=R2,transpose=True)
                elif 'matmul' in name and 'quantizer' not in name:
                    # module.register_parameter("R3",R3.weight)
                    # module.register_parameter("R4",R4.weight)
                    continue
                else:
                    continue
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                if parent_name:  
                    parent = dict(model.named_modules())[parent_name]
                    setattr(parent, name.split('.')[-1], new_module)
                else:  
                    setattr(model, name, new_module)

        substitute_layers(model)

    set_quant_state(model,weight_quant=False,act_quant=False)

    
    if matmul_scale_type == "S3":
        act_scales = get_matmul_act_scales_S3(model, data_loader_val,num_samples)
    elif matmul_scale_type == "S4":
        act_scales = get_matmul_act_scales_S4(model, data_loader_val,num_samples)
    else:
        raise NotImplementedError
    save_path = os.path.join(output_path,f'{net_name}_matmul_scale_'+matmul_scale_type+'.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)


def mamba2d_classify_generate_matmul_scale():
    from mmengine.config import Config, ConfigDict, DictAction
    from mmengine.runner import Runner
    from model_image_classification.src.mamba import Mamba2DModel
    from model_image_classification.utils.datasets import build_dataset
    args = parse_args()
    model_cfg = './model_image_classification/config/mamba2d_b.py'
    cfg = Config.fromfile(model_cfg)
    cfg.model_ckpt= "./ckpt/mamba2d_b.pth"
    cfg.work_dir = './work_dirs/mamba2d'
    runner = Runner.from_cfg(cfg)
    
    runner._test_loop = runner.build_test_loop(runner._test_loop)  # type: ignore

    runner.call_hook('before_run')

    # make sure checkpoint-related hooks are triggered after `before_run`
    runner.load_or_resume()
    runner.hooks[1]._swap_ema_parameters()
    
    from model_image_classification.utils.utils import convert_vim_2_vim_torch
    convert_vim_2_vim_torch(runner.model.backbone,"cuda")
    output_path = "model_image_classification/ckpt/"
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
    


    w_cfg = {"dynamic_method":"per_tensor","n_bits":8}
    a_cfg = {"dynamic_method":"per_tensor","n_bits":8}
    from model_image_classification.utils.normalized_modules import MatMul
    from quantize import QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d
    from quantize.utils import set_quant_state
    def replace_layers(model, target_class, replacement_class):
        for name, child in model.named_children():
            # if 'matmul' in name:
            #     continue
            if isinstance(child, target_class):
                # Replace the layer with the new quantized version
                if target_class == MatMul:
                    setattr(model, name, replacement_class(x1_quant_params=w_cfg,x2_quant_params=a_cfg))
                else:
                    setattr(model, name, replacement_class(child,weight_quant_params=w_cfg,act_quant_params=a_cfg))
            else:
                # Recursively call this function on the child module
                replace_layers(child, target_class, replacement_class)
    
    replace_layers(runner.model, MatMul, QuantMatMul)
    set_quant_state(runner.model,weight_quant=False,act_quant=False)
    
    act_scales = get_matmul_act_scales(runner.model, data_loader_val,num_samples)
    save_path = os.path.join(output_path,f'{net_name}_matmul_scale.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)

def mamba_llm_generate_matmul_scale_S3(model):
    import torch,datasets,random
    traindata = datasets.load_dataset("hellaswag",split='test')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    trainenc = tokenizer("\n\n".join(traindata['ctx']), return_tensors='pt') 
    dataloader = []
    num_samples = 128
    for _ in range(num_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - 2048 - 1)
        j = i + 2048
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        dataloader.append((inp, tar))
    
    act_scales = {}    
    from quantize.int_matmul import QuantMatMul
    def stat_tensor(name, tensor):
        tmp = tensor.permute(0,1,3,2) #
        # tmp = tensor
        hidden_dim = tmp.shape[-1]
        tmp = tmp.reshape(-1, hidden_dim).abs().detach()
        # comming_max = torch.max(tmp, dim=0)[0].float().cpu()
        comming_max = torch.quantile(tmp.float(), 0.999999, dim=0).float().cpu()
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
        if isinstance(m, QuantMatMul):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    input = torch.cat([i[0] for i in dataloader],dim=0)[:2]
    model(input.to(model.device))

    for h in hooks:
        h.remove()
        
    return act_scales

def mamba_llm_generate_matmul_scale_S4(model):
    import torch,datasets,random
    traindata = datasets.load_dataset("hellaswag",split='test')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    trainenc = tokenizer("\n\n".join(traindata['ctx']), return_tensors='pt') 
    dataloader = []
    num_samples = 128
    for _ in range(num_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - 2048 - 1)
        j = i + 2048
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        dataloader.append((inp, tar))
    
    act_scales = {}    
    from quantize.int_matmul import QuantMatMul
    def stat_tensor(name, tensor):
        tmp = tensor.permute(0,1,3,2) #
        # tmp = tensor
        hidden_dim = tmp.shape[-1]
        tmp = tmp.reshape(-1, hidden_dim).abs().detach()
        # comming_max = torch.max(tmp, dim=0)[0].float().cpu()
        comming_max = torch.quantile(tmp.float(), 0.999999, dim=0).float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[1]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, QuantMatMul):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    input = torch.cat([i[0] for i in dataloader],dim=0)[:2]
    model(input.to(model.device))

    for h in hooks:
        h.remove()
        
    return act_scales



if __name__ == '__main__':
    vim_generate_matmul_scale()
    # mamba2d_classify_generate_matmul_scale()
