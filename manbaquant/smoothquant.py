import torch
import torch.nn as nn

def vim_mambablock_smootquant(layers,act_scales):
    for i,layer in enumerate(layers):
        smooth_ln_fcs(layer.norm, layer.mixer.in_proj, act_scales[f"layers.{i}.mixer.in_proj"])

def mamband_mambablock_smootquant(layers,act_scales):
    for i,layer in enumerate(layers):
        smooth = smooth_ln_fcs(layer.norm, layer.mixer.in_proj, act_scales[f"backbone.layers.{i}.mixer.in_proj"])
        layer.smooth = smooth

def mamband_vit_mambablock_smootquant(layers,act_scales):
    for i,layer in enumerate(layers):
        smooth = smooth_ln_fcs(layer.norm, layer.mixer.in_proj, act_scales[f"vit.layers.{i}.mixer.in_proj"])
        layer.smooth = smooth

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    # assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)
    ln.weight.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
        
    return scales

