from collections import OrderedDict
from .int_linear import QuantLinear
from .int_conv import QuantConv1d,QuantConv2d
import torch
from .int_matmul import QuantMatMul
import random
import numpy as np
import logging
from datetime import datetime
import os
import sys
from .quantizer import UniformAffineQuantizer

class NoHookContext:
    def __init__(self, module):
        self.module = module
        self.hooks = []

    def __enter__(self):
        # 保存hooks
        for hook_id in list(self.module._forward_hooks.keys()):
            self.hooks.append((hook_id, self.module._forward_hooks.pop(hook_id)))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复hooks
        for hook_id, hook in self.hooks:
            self.module._forward_hooks[hook_id] = hook

class Logger(object):
    def __init__(self, folder="logs"):
        # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 创建日志文件夹（如果不存在）
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # 定义日志文件名
        filename = os.path.join(folder, f"log_{current_time}.txt")
        
        # 打开日志文件
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()


def set_seed(seed):
    torch.manual_seed(seed)  # 设置 CPU 上的随机数种子
    torch.cuda.manual_seed(seed)  # 设置当前 GPU 上的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 上的随机数种子（如果有多个 GPU）
    np.random.seed(seed)  # 设置 NumPy 的随机数种子
    random.seed(seed)  # 设置 Python 自带的随机数种子

    # 如果使用了 CuDNN 后端
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 确保卷积算法的选择是确定的
def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )



def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for name,m in self.named_modules():
        if isinstance(m, (QuantLinear, QuantMatMul,QuantConv1d,QuantConv2d)):
            m.set_quant_state(weight_quant, act_quant)

def set_static_quant(self, static_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    for m in self.modules():
        if isinstance(m, UniformAffineQuantizer):
            m.is_dynamic_quant = not static_quant

def set_static_quant_weight(self, static_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    for name, m in self.named_modules():
        if "weight" in name:
            if isinstance(m, UniformAffineQuantizer):
                m.is_dynamic_quant = not static_quant

def set_observing(self, observing: bool = True):
    self.use_observing = observing
    for name, m in self.named_modules():
        if isinstance(m, (UniformAffineQuantizer)):
           m.is_observing = observing
