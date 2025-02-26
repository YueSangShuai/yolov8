import torch,torch.nn as nn,torch.nn.functional as F
from .quantizer import UniformAffineQuantizer



# C8C8Add
class QuantAdd(nn.Module):
    def __init__(self,
                 x1_quant_params: dict = {},
                 x2_quant_params: dict = {},
                 ):
        super().__init__()
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        self.use_act_quant = False
    
    def forward(self,x1,x2):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
            x2 = self.x2_quantizer(x2)
        return x1 + x2
    

class QuantSoftmax(nn.Module):
    def __init__(self,act_quant_params:dict = dict(),dim=-1):
        super().__init__()
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.dim = dim
        self.use_act_quant = False
    
    def forward(self,attn_weights,attention_mask=None):
        ret_dtype = attn_weights.dtype
        if self.use_act_quant:
            attn_weights = self.act_quantizer(attn_weights)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        return F.softmax(attn_weights,dim=-1,dtype=torch.float32).to(ret_dtype)

class QuantSwiglu(nn.Module):
    def __init__(self,x1_quant_params=dict(),x2_quant_params = dict()):
        super().__init__()
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        self.smooth = None #  控制x*sigmoid(x)中的平滑系数
        self.use_act_quant = False
        # self.register_buffer("smooth",torch.ones)

    def forward(self,x1,x2):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
            x2 = self.x2_quantizer(x2)
        if self.smooth is  None:
            return x1 *  F.sigmoid(x1) * x2
        else:
            return x1 * F.sigmoid(x1 / self.smooth.to(x1.device)) * x2
        
class QuantSwilu(nn.Module):
    def __init__(self,x1_quant_params=dict(),x2_quant_params = dict()):
        super().__init__()
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        self.smooth = None #  控制x*sigmoid(x)中的平滑系数
        self.use_act_quant = False
        # self.register_buffer("smooth",torch.ones)

    def forward(self,x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        if self.smooth is  None:
            return x1 *  F.sigmoid(x1)
        else:
            return x1 * F.sigmoid(x1 * self.smooth.to(x1.device).view(1,1,-1))