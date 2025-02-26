import os
from pickle import FALSE, TRUE
import sys

# from model_image_classification.tools import train
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sympy import false
import torch,datasets
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_py.modeling_mamba import *
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate
import argparse
from datetime import datetime
import random

from quantize.utils import set_seed,Logger
set_seed(10)

# 将 sys.stdout 重定向到 Logger 类实例
folder = 'logs'
sys.stdout = Logger(folder=folder)

PLOT_DATA=False
USE_SMOOTHQUANT = False
USE_GPTQ = False
USE_HADMARD=False
USE_HADMARD_R1 = False
USE_HADMARD_R3S = False
USE_PERKERNEl=False
QUANT_WEIGHT = False
QUANT_ACT = False
w_bit = 8
a_bit = 8
w_cfg = {"dynamic_method":"per_tensor","n_bits":w_bit}
# w_cfg = {"dynamic_method":"per_channel","per_channel_axes":[0],"n_bits":4}
if USE_PERKERNEl:
    conv1d_w_cfg = {"dynamic_method":"per_channel","per_channel_axes":[2],"n_bits":w_bit}
else:
    conv1d_w_cfg = w_cfg
    
a_cfg = {"dynamic_method":"per_tensor","n_bits":a_bit}

from quantize import QuantConv1d,QuantConv2d,QuantLinear,QuantMatMul
from quantize.plot_utils.utils import plot_line_fig,plot_quantile_fig,plot_box_data_perchannel_fig, plot_bar_fig, plot_bar3d_fig,concat_images,find_images

def resize_tensor(tensor, size=(500, 500)):
    """
    将二维 tensor 重新采样到指定大小。

    Args:
        tensor: 输入的二维 tensor。
        size: 输出的大小，默认为 (500, 500)。

    Returns:
        重新采样后的 tensor。
    """
    h, w = tensor.shape
    target_h, target_w = size

    # 如果某个维度小于目标大小，不进行降采样
    new_h = min(h, target_h)
    new_w = min(w, target_w)
    
    # 调整到新尺寸
    resized_tensor = F.interpolate(tensor.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='nearest')
    
    # 将 tensor 的维度恢复
    return resized_tensor.squeeze(0).squeeze(0)

def analyse_hook(module, input, output): 
    module_name = module_to_name.get(module, "Unnamed module")
    os.makedirs(f"data/analyse_fig/{input_name}/{quant_name}/", exist_ok=True)
    save_dir = f"data/analyse_fig/{input_name}/{quant_name}"
    
    # 分析权重
    if isinstance(module, (QuantConv2d,QuantLinear)):
        weight = module.weight.data
        print(module_name,"_weight  shape: ",weight.shape)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg", axis=1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg", axis=0)
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg"):
        #     plot_box_data_perchannel_fig(weight, f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg", axis=-1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_bar_data.jpg"):
            plot_bar_fig(weight, f"{save_dir}/{module_name}_weight_bar_data.jpg")
        if not os.path.exists(f"{save_dir}/{module_name}_weight_bar3d_data.jpg"):
            plot_bar3d_fig(resize_tensor(weight), f"{save_dir}/{module_name}_weight_bar3d_data.jpg")
    
    # 分析权重
    if isinstance(module, QuantConv1d):
        weight = module.weight.data[:,0]
        print(module_name,"_weight  shape: ",weight.shape)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg", axis=1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg", axis=0)
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg"):
        #     plot_box_data_perchannel_fig(weight, f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg", axis=-1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_bar_data.jpg"):
            plot_bar_fig(weight, f"{save_dir}/{module_name}_weight_bar_data.jpg")
        if not os.path.exists(f"{save_dir}/{module_name}_weight_bar3d_data.jpg"):
            plot_bar3d_fig(resize_tensor(weight), f"{save_dir}/{module_name}_weight_bar3d_data.jpg")

    for i,temp_input in enumerate(input):
        if isinstance(temp_input, torch.Tensor):  # 确保有输入且为Tensor类型
            print(module_name,f"_input_{i}  shape: ",temp_input.shape)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_token_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_token_quantile.jpg", axis=1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg", axis=2)
            # if not os.path.exists(f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg"):
            #     plot_box_data_perchannel_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg", axis=-1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar_data.jpg"):
                plot_bar_fig(temp_input, f"{save_dir}/{module_name}_input{i}_bar_data.jpg")
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg"):
                plot_bar3d_fig(resize_tensor(torch.amax(temp_input,dim=0)), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")


def analyse_hook_2(module, input, output):  # matmul画图
    module_name = module_to_name.get(module, "Unnamed module")
    os.makedirs(f"data/analyse_fig/{input_name}/{quant_name}/", exist_ok=True)
    save_dir = f"data/analyse_fig/{input_name}/{quant_name}"
    
    for i,temp_input in enumerate(input):
        if isinstance(temp_input, torch.Tensor):  # 确保有输入且为Tensor类型
            print(module_name,f"_input_{i}  shape: ",temp_input.shape)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_token_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_token_quantile.jpg", axis=1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_quantile.jpg", axis=2)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg"):
                plot_box_data_perchannel_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg", axis=-1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar_data.jpg"):
                plot_bar_fig(temp_input, f"{save_dir}/{module_name}_input{i}_bar_data.jpg")
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg"):
                plot_bar3d_fig(resize_tensor(torch.amax(temp_input,dim=0)), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")
    

def register_hooks(model):
    global module_to_name 
    global quant_name
    quant_name = "fp_data"
    module_to_name = {module: name for name, module in model.named_modules()}
    handles = []
    for i,layer in enumerate(model):

        # 新增后处理hook
        handles.append(layer.mixer.conv1d.register_forward_hook(analyse_hook))
        handles.append(layer.mixer.in_proj.register_forward_hook(analyse_hook))
        handles.append(layer.mixer.x_proj.register_forward_hook(analyse_hook))
        handles.append(layer.mixer.dt_proj.register_forward_hook(analyse_hook))
        handles.append(layer.mixer.out_proj.register_forward_hook(analyse_hook))
        # handles.append(layer.mixer.conv_matmul.register_forward_hook(analyse_hook_2))
        handles.append(layer.mixer.ssm_matmul.register_forward_hook(analyse_hook_2))
    return handles

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


@register_model("mamba")
class MambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="/data01/home/xuzk/datasets/lm_mamba_weight/mamba-130m-hf", max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.float16,add_bos_token=False):
        super().__init__(pretrained)
        LM.__init__(self)
        from mamba_py.modeling_mamba import MambaForCausalLM
        self._model = MambaForCausalLM.from_pretrained(pretrained)
        # self._model = AutoModelForCausalLM.from_pretrained(pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)
        self.add_bos_token = add_bos_token

        self._model.half().to(device)



        # from utils.utils import convert_vim_2_vim_torch
        from mamba_py.normalized_modules import MatMul
        from quantize import QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d
        # convert_vim_2_vim_torch(self._model.backbone,device)
        def replace_layers(model, target_class, replacement_class):
            for name, child in model.named_children():
                if isinstance(child, target_class):
                    # Replace the layer with the new quantized version
                    if target_class == MatMul:
                        setattr(model, name, replacement_class(x1_quant_params=a_cfg,x2_quant_params=a_cfg))
                    elif "conv1d" in name:
                        setattr(model, name, replacement_class(child,weight_quant_params=conv1d_w_cfg,act_quant_params=a_cfg))
                    else:
                        setattr(model, name, replacement_class(child,weight_quant_params=w_cfg,act_quant_params=a_cfg))
                else:
                    # Recursively call this function on the child module
                    replace_layers(child, target_class, replacement_class)

        replace_layers(self._model, MatMul, QuantMatMul)
        replace_layers(self._model, nn.Linear, QuantLinear)
        replace_layers(self._model, nn.Conv1d, QuantConv1d)
        replace_layers(self._model, nn.Conv2d, QuantConv2d)
        from quantize.utils import set_quant_state
        set_quant_state(self._model,weight_quant=QUANT_WEIGHT,act_quant=QUANT_ACT)

        from quantize.hm_model_utils import fuse_layer_norms, RotateModule, RQuantLinear
        from quantize.hadmard import random_hadamard_matrix
        h3 = random_hadamard_matrix(self._model.backbone.layers[0].mixer.out_proj.in_features,device)
        R3 = RotateModule(h3)
        # matmul_scale = torch.load("saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_matmul_scale.pt",map_location=device)


        if USE_HADMARD:
            if USE_HADMARD_R3S:
                def substitute_layers(model):
                    
                    for name,module in model.named_modules():
                        if 'ssm_matmul' in name and 'quantizer' not in name:    
                            # module.register_parameter("matmul_scale",torch.nn.Parameter(matmul_scale[name]))
                            module.register_parameter("R3",R3.weight)
                            # module.register_parameter("R4",R4.weight)
                            continue
                        else:
                            continue

                substitute_layers(self._model)   

            if USE_HADMARD_R1:
                h1 = random_hadamard_matrix(self._model.backbone.layers[0].mixer.in_proj.in_features,device)
                R1 = RotateModule(h1)
                fuse_layer_norms(self._model.backbone)
                def substitute_R1_layers(model):
                    for name,module in model.named_modules():
                        if 'in_proj' in name and 'quantizer' not in name:
                            new_module = RQuantLinear(module,R1=R1,transpose=False)
                        elif 'out_proj' in name and 'quantizer' not in name:
                            new_module = RQuantLinear(module,R1=R1,transpose=True)
                        else:
                            continue
                        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                        if parent_name:  
                            parent = dict(model.named_modules())[parent_name]
                            setattr(parent, name.split('.')[-1], new_module)
                        else:  
                            setattr(model, name, new_module)

                substitute_R1_layers(self._model) 
                self._model.backbone.register_parameter("R1",R1.weight)   
                set_quant_state(self._model,weight_quant=True,act_quant=True) 

        if USE_GPTQ:
            if os.path.exists(pretrained.split("/")[-1]+"_gptq_{}bit.pt".format(args.w_cfg['n_bits'])):
                self._model.load_state_dict(torch.load(pretrained.split("/")[-1]+"_gptq_{}bit.pt".format(args.w_cfg['n_bits'])))
            else:
                import argparse
                args = argparse.Namespace()
                args.seqlen = 2048
                args.nsamples = 128
                args.w_bits=w_bit
                traindata = datasets.load_dataset("hellaswag",split='test')
                trainenc = self.tokenizer("\n\n".join(traindata['ctx']), return_tensors='pt') 
                trainloader = []
                for _ in range(args.nsamples):
                    i = random.randint(0, trainenc.input_ids.shape[1] - args.seqlen - 1)
                    j = i + args.seqlen
                    inp = trainenc.input_ids[:, i:j]
                    tar = inp.clone()
                    tar[:, :-1] = -100
                    trainloader.append((inp, tar))

                from quantize.gptq import gptq_fwrd_llm
                gptq_fwrd_llm(self._model,trainloader,dev='cuda',args=args)
                self._model.to(device)
                torch.save(self._model.state_dict(),pretrained.split("/")[-1]+"_gptq_{}bit.pt".format(args.w_cfg['n_bits']))

        if USE_SMOOTHQUANT:

            if not os.path.exists("mamba_llm_shift.pt"):
                traindata = datasets.load_dataset("hellaswag",split='test')
                trainenc = self.tokenizer("\n\n".join(traindata['ctx']), return_tensors='pt') 
                trainloader = []
                for _ in range(128):
                    i = random.randint(0, trainenc.input_ids.shape[1] - 2048 - 1)
                    j = i + 2048
                    inp = trainenc.input_ids[:, i:j]
                    tar = inp.clone()
                    tar[:, :-1] = -100
                    trainloader.append((inp, tar))

                from quantize.smoothquant_generate_act_scale_shift import mamba_llm_generate_act_scale_shift
                mamba_llm_generate_act_scale_shift(self._model,trainloader)
            act_scales = torch.load("mamba_llm_shift.pt")
            device,dtype = next(self._model.parameters()).device,next(self._model.parameters()).dtype
            def smooth_ln_fcs(model,act_scales,alpha=0.5):
                for name, module in model.named_modules():
                    if isinstance(module,nn.Linear) and 'in_proj' in name:
                        weight_scale = module.weight.max(dim=0)[0].clamp(min=1e-5)
                        act_scale = act_scales[name].to(device)
                        scales = ((act_scale.pow(alpha)/weight_scale.pow(1-alpha)).clamp(min=1e-5)).to(device).to(dtype)
                        with torch.no_grad():
                            module.weight.div_(scales)

                        parts = name.split('.')
                        parent_name = '.'.join(parts[:-2])
                        rms_name = parent_name + '.norm'
                        for subname, submodule in model.named_modules():
                            if subname == rms_name:
                                with torch.no_grad():
                                    submodule.weight.mul_(scales)


            smooth_ln_fcs(self._model,act_scales)

        if PLOT_DATA:
            global input_name
            input_name = "fig"
            seqlen = 258
            nsamples = 64
            traindata = datasets.load_dataset("hellaswag",split='test', download_mode="reuse_cache_if_exists")
            trainenc = self.tokenizer("\n\n".join(traindata['ctx']), return_tensors='pt') 
            inps = []
            for _ in range(nsamples):    
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
                inps.append(inp)
            inp = torch.concat(inps, dim=0).to(self._device)
            handles = register_hooks(self._model.backbone.layers)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    self._model(inp)
            for handle in handles:handle.remove() 
            
            # if os.path.exists(f"data/analyse_fig/{input_name}/{quant_name}/"):
            #     suffixes = count_suffixes(f"data/analyse_fig/{input_name}/{quant_name}/")
            #     for ss in suffixes.keys():
            #         image_paths = find_images(f"data/analyse_fig/{input_name}/{quant_name}/", "", ss)
            #         filter_images = [image for image in image_paths if image.split(".")[1].isdigit()]
            #         sorted_images = sorted(filter_images,key=lambda x: int(x.split(".")[1]))
            #         os.makedirs(f"data/analyse_fig/{input_name}/{quant_name}_cat/", exist_ok=True)
            #         concat_images(sorted_images, 6, f"data/analyse_fig/{input_name}/{quant_name}_cat/{ss}")
            

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Naive Script")
    parser.add_argument('--PLOT_DATA', type=bool, default=True)
    parser.add_argument('--USE_SMOOTHQUANT', type=bool, default=False, help='Use SmoothQuant')
    parser.add_argument('--USE_GPTQ', type=bool, default=False, help='Use GPTQ')
    parser.add_argument('--USE_HADMARD', type=bool, default=False, help='Use Hadmard')
    parser.add_argument('--USE_HADMARD_R1', type=bool, default=False, help='Use Hadmard R1')
    parser.add_argument('--USE_HADMARD_R3S', type=bool, default=False, help='Use Hadmard R3S')
    parser.add_argument('--USE_PERKERNEl', type=bool, default=False, help='Use PerKernel')
    parser.add_argument('--QUANT_WEIGHT', type=bool, default=False)
    parser.add_argument('--QUANT_ACT', type=bool, default=False)
    parser.add_argument('--w_bit', type=int, default=4)
    parser.add_argument('--a_bit', type=int, default=8)
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args
    
    QUANT_WEIGHT = args.QUANT_WEIGHT
    QUANT_ACT = args.QUANT_ACT
    w_bit = args.w_bit
    a_bit = args.a_bit
    w_cfg = {"dynamic_method":"per_tensor","n_bits":w_bit}
    # w_cfg = {"dynamic_method":"per_channel","per_channel_axes":[0],"n_bits":4}
    if USE_PERKERNEl:
        conv1d_w_cfg = {"dynamic_method":"per_channel","per_channel_axes":[2],"n_bits":w_bit}
    else:
        conv1d_w_cfg = w_cfg
        
    a_cfg = {"dynamic_method":"per_tensor","n_bits":a_bit}
    PLOT_DATA = args.PLOT_DATA
    USE_SMOOTHQUANT = args.USE_SMOOTHQUANT
    USE_GPTQ = args.USE_GPTQ 
    USE_HADMARD = args.USE_HADMARD
    USE_HADMARD_R1 = args.USE_HADMARD_R1
    USE_HADMARD_R3S = args.USE_HADMARD_R3S
    USE_PERKERNEl = args.USE_PERKERNEl 
    print(args)
    print('PLOT_DATA',PLOT_DATA)
    print('QUANT_WEIGHT',QUANT_WEIGHT)
    print('QUANT_ACT',QUANT_ACT)
    print("w_cfg  " , w_cfg)
    print("a_cfg  " , a_cfg)
    print("USE_SMOOTHQUANT ",USE_SMOOTHQUANT)
    print("USE_GPTQ  ",USE_GPTQ)
    print('USE_HADMARD  ',USE_HADMARD)
    print('USE_HADMARD_R1  ',USE_HADMARD_R1)
    print('USE_HADMARD_R3S  ',USE_HADMARD_R3S)
    print('USE_PERKERNEl  ',USE_PERKERNEl)
    print()

    cli_evaluate()


