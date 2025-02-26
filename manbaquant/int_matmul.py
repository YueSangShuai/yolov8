from numpy import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import UniformAffineQuantizer


class QuantMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict = {"dynamic_method":"per_tensor"},
        x2_quant_params: dict = {"dynamic_method":"per_tensor"},
        disable_act_quant=False,
        observe = "minmax",
        matmul_func=torch.matmul,
    ):
        super().__init__()
        # de-activate the quantized forward default
        self.use_act_quant = False
        # initialize quantizer
        self.i_cluster_counts = None
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params,has_batch_dim=True,observe=observe)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params,has_batch_dim=True,observe=observe)
        self.matmul_func = matmul_func

        self.disable_act_quant = disable_act_quant


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2):
        if self.use_act_quant:
            x2 = self.x2_quantizer(x2)
        return x2

    def forward(self, x1, x2):
        if hasattr(self,"pertoken"):
            B,L,ED,N = x1.shape
            x1 = x1.reshape(B,L*ED,N)
            x1 = self.quant_x1(x1)
            x1 = x1.reshape(B,L,ED,N)
            x2 = self.quant_x2(x2)
            out = self.matmul_func(x1, x2)
            pass
        else:
            x1 = self.quant_x1(x1)
            x2 = self.quant_x2(x2)
            out = self.matmul_func(x1, x2)
        return out
