import torch
from torch import nn
import typing
from quantize.int_linear import QuantLinear

def fuse_mamband_layer_norms(model):
   
    layers = model.layers
    for layer in layers:  
        with torch.no_grad(): 
            norm = layer.norm
            ln_in = layer.mixer.in_proj
            ln_out= layer.mixer.out_proj
            
            
            
            
            
            layer.norm.weight.fill_(1.)

def fuse_layer_norms(model):
   
    layers = model.layers
    for layer in layers:
        fuse_ln_linear(layer.norm, [layer.mixer.in_proj])   
        with torch.no_grad(): 
            layer.norm.weight.fill_(1.)

def fuse_layer_norms_2(model):
   
    layers = model.layers
    for layer in layers:
        fuse_ln_linear(layer.norm, [layer.mixer.in_proj_states,layer.mixer.in_proj_gates])   
        with torch.no_grad(): 
            layer.norm.weight.fill_(1.)

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias') and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)


class RotateModule(nn.Module):
    def __init__(self, R_init):
        super().__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32).to(torch.device("cuda")))

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x

class RQuantLinear(QuantLinear):
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
        R1:nn.Module = None,
        R2:nn.Module = None,
        transpose=False
    ):
        super().__init__(org_module, weight_quant_params, act_quant_params, disable_input_quant)
        self.R1 = R1
        self.R2 = R2
        self.transpose = transpose
        self.rotated_flag = False

    def forward(self, input: torch.Tensor):
        if not self.rotated_flag:
            weight = self.weight
            dtype = weight.dtype
            if self.transpose:
                weight = (self.R1.weight.T.to(torch.float64)@weight.to(torch.float64)).to(dtype)
                # weight = (weight.to(torch.float64)@self.R2.weight.to(torch.float64)).to(dtype)
            else:
                weight = (weight.to(torch.float64)@self.R1.weight.to(torch.float64)).to(dtype)
            self.rotated_flag = True
            self.weight = torch.nn.Parameter(weight)


        if self.use_weight_quant:
            self.weight.data = self.weight_quantizer(self.weight)

        
        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        if self.bias is not None:self.bias = self.bias.to(self.weight)
        out = self.fwd_func(
                input.to(self.weight), self.weight, self.bias, **self.fwd_kwargs)
        
    
        return out

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output   
