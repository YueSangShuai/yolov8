import sys
import math
import time
import tqdm
import torch
import torch.nn as nn
from quantize import utils
import logging

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = -maxq -1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def two_compl(x, bits: int):
    return torch.where(x < 0, 2 ** bits + x, x)

# Pack the int tensor. Each uint8 stores two int4 value.
def pack_i4(q):
    assert torch.is_signed(q), 'The tensor to be packed should be signed int'
    minq, maxq = get_minq_maxq(4, True)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4

def unpack_i4(x: torch.Tensor):
    assert x.dtype == torch.uint8, 'The tensor to be unpacked should be stored in uint8'

    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # Low 4 bits
    x0 = (x & 0x0f).to(torch.int8)
    x0[x0>=8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # High 4 bits
    x1 = ((x & 0xf0) >> 4).to(torch.int8)
    x1[x1>=8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # Interleaving
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)


class ActQuantizer(torch.nn.Module):

    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16

    def free(self):
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16: # 初始用16然后多次进行配置
            return x
        elif self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(self, bits, groupsize=-1, sym=False, clip_ratio=1.0,act_quant='per-tensor'):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.act_quant = act_quant
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params_per_tensor(self, x):
        # init_shape = x.shape
        # reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // 1, 1)
        xmax = torch.amax(x) * self.clip_ratio
        xmin = torch.amin(x) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            # tmp = xmax == 0
            self.scale = xmax / self.maxq
            if xmax==0:
                self.scale = torch.tensor(1).to(xmax.device).to(xmax.dtype)
            self.zero = torch.zeros_like(self.scale)
        else:
            print ('exist asym act')
            sys.exit()

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            utils.cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registerd to rotate the activation before quantization.
    '''

    def __init__(self, module:torch.nn.Linear):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

        return str_

    def forward(self, x):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had: # down_proj had_k不规则11122, weight也会做一次来匹配
            
            if self.fp32_had: # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else: # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
            
        elif self.online_partial_had: # o_proj  
            # todo: implement this in QAttention to avoid reshaping!
            
            if self.fp32_had:
                x = x.float()
                
            init_shape = x.shape
            if self.K == 1: # 走这个分支,由于v的weight再head_dim维上做了一个H 因此抵消掉了
                # b,l,c -> bl,h,d -> bl,d,h
                x = fast_hadamard_transform.hadamard_transform(x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim).transpose(1, 2),
                                                               scale=1/math.sqrt(init_shape[-1]//self.had_dim)).transpose(1, 2)
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1]//self.had_dim)
                
            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16: #Quantize, if needed
            if self.quantizer.act_quant == 'per-tensor':
                self.quantizer.find_params_per_tensor(x)
            elif self.quantizer.act_quant == 'per-token':
                self.quantizer.find_params(x)
            else:
                print ('quantizer not support !!!')
                sys.exit()
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16: #Quantize the output, if needed
            if self.out_quantizer.act_quant =='per-tensor':
                self.out_quantizer.find_params_per_tensor(x)
            elif self.out_quantizer.act_quant =='per-token':
                self.out_quantizer.find_params(x)
            else:
                print ('out_quantizer not support !!!')
                sys.exit()
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x

class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,pot=False
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)
        self.pot = pot

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            # xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5) # TODO 
            # self.scale = xmax / self.maxq
            self.scale = torch.maximum((xmin.clip(max=0))/(-self.maxq-1) ,xmax.clip(min=1e-6)/(self.maxq))
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = torch.maximum((xmin1.clip(max=0))/(-self.maxq-1) ,xmax1.clip(min=1e-6)/(self.maxq))
                    # scale1 = xmax1 / self.maxq
                    if self.pot:
                        scale1 = 2**(scale1.log2().ceil())
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)



def find_qlayers(module, layers=[torch.nn.Linear,
                                ActQuantWrapper], name=''):
    if isinstance(module,tuple(layers)):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)
        
        
@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    sequential = [
                ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
                ['self_attn.o_proj.module'],
                ['mlp.up_proj.module', 'mlp.gate_proj.module'],
                ['mlp.down_proj.module']
            ]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers

def gptq_fwrd_vim(model, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    layers = model.layers

    model.patch_embed = model.patch_embed.to(dev)
    model.norm_f = model.norm_f.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples,dataloader.batch_size, model.patch_embed.num_patches+1, model.embed_dim), dtype=dtype, device=dev
    )
    inps_2 = torch.zeros(
        (args.nsamples,dataloader.batch_size, model.patch_embed.num_patches+1, model.embed_dim), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp,residual, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            # cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i, batch in enumerate(dataloader):
        if i == args.nsamples:
            break
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.layers[0] = layers[0]

    layers[0] = layers[0].cpu()
    model.patch_embed = model.patch_embed.cpu()
    model.norm_f = model.norm_f.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    outs_2 = torch.zeros_like(inps_2)
    # attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']

    quantizers = {}
    sequential = [
                # ['mixer.in_proj', 'mixer.x_proj', 'mixer.dt_proj', 'mixer.x_proj_b', 'mixer.dt_proj_b', 'mixer.out_proj']
                ['mixer.in_proj_states', 'mixer.in_proj_gates',
                 'mixer.x_proj_C',"mixer.x_proj_B","mixer.x_proj_dt", 'mixer.dt_proj', 
                 'mixer.x_proj_C_b',"mixer.x_proj_B_b","mixer.x_proj_dt_b", 'mixer.dt_proj_b', 
                 'mixer.out_proj']
            ]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = False
                # if 'lm_head' in name:
                #     layer_weight_bits = 16
                #     continue
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=args.perchannel, sym=layer_weight_sym, mse=True
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            with torch.no_grad():    
                for j in range(args.nsamples):
                    hidden,resual = layer(inps[j].cuda(),inps_2[j].cuda())
                    outs[j],outs_2[j] = hidden.cpu(),resual.cpu()
                    del hidden,resual
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        with torch.no_grad():  
            for j in range(args.nsamples):
                hidden,resual = layer(inps[j].cuda(),inps_2[j].cuda())
                outs[j],outs_2[j] = hidden.cpu(),resual.cpu()
                del hidden,resual

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps,inps_2, outs,outs_2 = outs, outs_2, inps, inps_2

    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers

def gptq_fwrd_mamba2d(model, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    backbone = model.backbone

    layers = backbone.layers

    backbone.patch_embed = backbone.patch_embed.to(dev)
    backbone.ln1 = backbone.ln1.to(dev)
    backbone.ln2 = backbone.ln2.to(dev)
    backbone.norm1 = backbone.norm1.to(dev)
    backbone.norm2 = backbone.norm2.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(backbone.parameters())).dtype
    inps = [1] * args.nsamples
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp,residual, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            # cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i, batch in enumerate(dataloader):
        if i == args.nsamples:
            break
        try:
            model.test_step(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    backbone.layers[0] = layers[0]

    layers[0] = layers[0].cpu()
    backbone.patch_embed = backbone.patch_embed.cpu()
    backbone.ln1 = backbone.ln1.cpu()
    backbone.ln2 = backbone.ln2.cpu()
    backbone.norm1 = backbone.norm1.cpu()
    backbone.norm2 = backbone.norm2.cpu()
    torch.cuda.empty_cache()

    outs = [1] * args.nsamples
    # attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']

    quantizers = {}
    sequential = [
                ['mixer.in_proj', 'mixer.x_proj', 'mixer.dt_proj', 'mixer.out_proj']
            ]
    

    residual = None
    h = h_ = backbone.patch_resolution[0]
    w = w_ = backbone.patch_resolution[1]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = True
                # if 'lm_head' in name:
                #     layer_weight_bits = 16
                #     continue
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=False, sym=layer_weight_sym, mse=True
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

  
            h_tmp,w_tmp = 28,28
            for j in range(args.nsamples):
                outs[j],residual = layer(inps[j],residual,h=h,w=w)
                if layer.downsample:
                    h_tmp,w_tmp=residual
                    residual = None

            if layer.downsample:
                h,w = h_tmp,w_tmp
                residual = None

            for handle in handles:
                handle.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        h_tmp,w_tmp = 28,28
        for j in range(args.nsamples):
            outs[j],residual = layer(inps[j],residual,h=h_,w=w_)
            if layer.downsample:
                h_tmp,w_tmp = residual
                residual = None
                
        if layer.downsample:
            h_,w_ = h_tmp,w_tmp
            residual = None

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps,outs = outs, inps

    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers

def gptq_fwrd_mamba3d(model, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    #先过一遍网络，为了网络中的一些hadamard旋转和smooth生效
    for i, batch in enumerate(dataloader):
        model.test_step(batch)
        break
    
    backbone = model.backbone

    layers = backbone.layers

    backbone.patch_embed = backbone.patch_embed.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(backbone.parameters())).dtype
    inps = [1] * args.nsamples
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp,**kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            # cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i, batch in enumerate(dataloader):
        if i == args.nsamples:
            break
        try:
            model.test_step(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    backbone.layers[0] = layers[0]

    layers[0] = layers[0].cpu()
    backbone.patch_embed = backbone.patch_embed.cpu()
    torch.cuda.empty_cache()

    outs = [1] * args.nsamples
    # attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']

    quantizers = {}
    sequential = [
                # ['mixer.in_proj', 'mixer.x_proj', 'mixer.dt_proj', 'mixer.out_proj']
                ["mixer.in_proj_states", "mixer.in_proj_gates", "mixer.x_proj_B","mixer.x_proj_C",
                 "mixer.x_proj_dt","mixer.dt_proj","mixer.out_proj"]
            ]
    
    order = torch.load("./work_dirs/order.pth")
    n_pos_dim = torch.load("./work_dirs/n_pos_dim.pth")
    shape = torch.load("./work_dirs/shape.pth")
    
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        
        full = find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = True
                # if 'lm_head' in name:
                #     layer_weight_bits = 16
                #     continue
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=False, sym=layer_weight_sym, mse=True
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            with torch.no_grad():  
                for j in range(args.nsamples):
                    outs[j] = layer(inps[j],order=order[i],shape=shape[i],n_dim_pos=n_pos_dim[i])

            for handle in handles:
                handle.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        with torch.no_grad():  
            for j in range(args.nsamples):
                outs[j] = layer(inps[j],order=order[i],shape=shape[i],n_dim_pos=n_pos_dim[i])

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps,outs = outs, inps

    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers

def gptq_fwrd_mamba3d_seg(model, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    backbone = model.keywords['predictor'].vit

    layers = backbone.layers

    backbone.patch_embed = backbone.patch_embed.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(backbone.parameters())).dtype
    inps = [1] * args.nsamples
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp,**kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            # cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i, batch in enumerate(dataloader):
        data = batch["image"].to(dev)
        if i == args.nsamples:
            break
        try:
            model(data)
        except ValueError:
            pass
    layers[0] = layers[0].module
    backbone.layers[0] = layers[0]

    layers[0] = layers[0].cpu()
    backbone.patch_embed = backbone.patch_embed.cpu()
    torch.cuda.empty_cache()

    outs = [1] * args.nsamples
    # attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']

    quantizers = {}
    sequential = [
                ['mixer.in_proj', 'mixer.x_proj', 'mixer.dt_proj', 'mixer.out_proj']
            ]
    

    
    order = torch.load("./runs/order.pth")
    n_pos_dim = torch.load("./runs/n_pos_dim.pth")
    shape = torch.load("./runs/shape.pth")
    
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = True
                # if 'lm_head' in name:
                #     layer_weight_bits = 16
                #     continue
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=False, sym=layer_weight_sym, mse=True
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            with torch.no_grad():  
                for j in range(args.nsamples):
                    outs[j] = layer(inps[j],order=order[i],shape=shape[i],n_dim_pos=n_pos_dim[i])

            for handle in handles:
                handle.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        with torch.no_grad():  
            for j in range(args.nsamples):
                outs[j] = layer(inps[j],order=order[i],shape=shape[i],n_dim_pos=n_pos_dim[i])

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps,outs = outs, inps

    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers

       
@torch.no_grad()
def rtn_fwrd(model, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = find_qlayers(layer,
                                            layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            quantizer = WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(
                next(iter(layer.parameters())).dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    utils.cleanup_memory(verbos=True)
    return quantizers

def gptq_fwrd_llm(model, dataloader, dev, args):
    
    logging.info('-----GPTQ Quantization-----')
    
    backbone = model.backbone

    layers = backbone.layers

    backbone.embeddings = backbone.embeddings.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(backbone.parameters())).dtype
    # inps = [1] * args.nsamples
    inps = torch.zeros(
         (args.nsamples, args.seqlen, layers[0].config.hidden_size), dtype=dtype, device=dev
     )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp,**kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            # cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i, batch in enumerate(dataloader):
        data = batch[0].to(dev)
        if i == args.nsamples:
            break
        try:
            model(data)
        except ValueError:
            pass
    layers[0] = layers[0].module
    backbone.layers[0] = layers[0]

    layers[0] = layers[0].cpu()
    backbone.embeddings = backbone.embeddings.cpu()
    torch.cuda.empty_cache()

    # outs = [1] * args.nsamples
    outs = torch.zeros_like(inps)
    # attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']

    quantizers = {}
    sequential = [
                # ['mixer.in_proj', 'mixer.x_proj', 'mixer.dt_proj', 'mixer.out_proj']
                ['mixer.in_proj_states', 'mixer.in_proj_gates',
                 'mixer.x_proj_c',"mixer.x_proj_b","mixer.x_proj_dt", 'mixer.dt_proj',
                 'mixer.out_proj']
            ]
    

    
    # order = torch.load("./runs/order.pth")
    # n_pos_dim = torch.load("./runs/n_pos_dim.pth")
    # shape = torch.load("./runs/shape.pth")
    
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = True
                # if 'lm_head' in name:
                #     layer_weight_bits = 16
                #     continue
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=True
                )
        

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            with torch.no_grad():
                for j in range(args.nsamples):
                    outs[j] = layer(inps[j].unsqueeze(0))[0]

            for handle in handles:
                handle.remove()

            for name in subset:
                layer_w_groupsize = -1
                gptq[name].fasterquant(
                    percdamp=.01, groupsize=layer_w_groupsize, actorder=False, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps,outs = outs, inps
        torch.cuda.empty_cache()

    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers
