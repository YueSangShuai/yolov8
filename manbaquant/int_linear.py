import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import UniformAffineQuantizer






class QuantLinear(nn.Linear):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {"dynamic_method":"per_tensor"},
        act_quant_params: dict = {"dynamic_method":"per_tensor"},
        disable_input_quant=False,
        observe = "minmax",
    ):
        super().__init__(org_module.in_features,org_module.out_features)
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.weight=org_module.weight
        if org_module.bias is not None:
            self.bias=org_module.bias
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape,is_weight=True,observe=observe)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,has_batch_dim=True,observe=observe)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        
        self.weight_quantized = False

    
    
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            if self.weight_quantizer.is_observing:
                weight = self.weight
            elif not self.weight_quantized:
                self.weight = torch.nn.Parameter(self.weight_quantizer(self.weight))
                weight = self.weight
                self.weight_quantized = True
            else:
                weight = self.weight
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)
        
        if bias is not None:bias = bias.to(weight)
        out = self.fwd_func(
                input.to(weight), weight, bias, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
