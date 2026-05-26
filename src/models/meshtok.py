from logging import getLogger
from functools import partial

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask
from einops import rearrange

from .attention_utils import (
    CustomTransformerEncoder,
    CustomTransformerEncoderLayer,
    CacheCustomTransformerEncoder,
    CacheCustomTransformerEncoderLayer,
    DynamicTanh,
)
from .embedder import get_embedder
from .kv_cache import KVCache


logger = getLogger()


def block_lower_triangular_mask(block_size, block_num, use_float=False):
    """
    Create a block lower triangular boolean mask. (upper right part will be 1s, and represent locations to ignore.)
    """
    matrix_size = block_size * block_num
    lower_tri_mask = torch.tril(torch.ones(matrix_size, matrix_size, dtype=torch.bool))
    block = torch.ones(block_size, block_size, dtype=torch.bool)
    blocks = torch.block_diag(*[block for _ in range(block_num)])
    final_mask = torch.logical_or(lower_tri_mask, blocks)

    if use_float:
        return torch.zeros_like(final_mask, dtype=torch.float32).masked_fill_(~final_mask, float("-inf"))
    else:
        return ~final_mask


def block_causal(b, h, q_idx, kv_idx, block_size):
    return (q_idx // block_size) >= (kv_idx // block_size) 


class MeshTok(nn.Module):
    def __init__(self, config, x_num, max_output_dim, max_data_len=1):
        super().__init__()
        self.config = config
        self.x_num = x_num
        self.max_output_dim = max_output_dim

    
        self.flex_attn = config.get("flex_attn", False)
        self.refine_ratio = config.get("refine_ratio", 0.125)
        self.per_refine = int(self.refine_ratio*config.patch_num*config.patch_num) 
        assert (
            abs(self.refine_ratio*config.patch_num*config.patch_num - self.per_refine)<1e-10
        ), f" patchs need to be int"

        self.embedder = get_embedder(config.embedder, x_num, max_output_dim,k=self.per_refine)

        match config.get("norm", "layer"):
            case "rms":
                norm = nn.RMSNorm
            case "dyt":
                norm = DynamicTanh
            case _:
                norm = nn.LayerNorm

        kwargs = {
            "all_exp": config.all_exp,
            "d_model": config.dim_emb,
            "nhead": config.n_head,
            "training": config.training,
            "topk": config.topk,
            "n_shared_experts": config.n_shared_experts,
            "moe_intermediate_size": config.moe_intermediate_size,
            "dim_feedforward": config.dim_ffn,
            "dropout": config.dropout,
            "attn_dropout": config.get("attn_dropout", 0),
            "activation": config.get("activation", "gelu"),
            "norm_first": config.norm_first,
            "norm": norm,
            "rotary": config.rotary,
            "qk_norm": config.get("qk_norm", False),
            "flex_attn": self.flex_attn,
        }

        if config.kv_cache:
            self.transformer = CacheCustomTransformerEncoder(
                model_type=CacheCustomTransformerEncoderLayer,
                kwarg=kwargs,
                is_dense=config.dense,
                num_layers=config.n_layer,
                norm=norm(config.dim_emb, eps=1e-5) if config.norm_first else None,
                config=config,
            )
        else:
            self.transformer = CustomTransformerEncoder(
                model_type=CustomTransformerEncoderLayer,
                kwarg=kwargs,
                is_dense=config.dense,
                num_layers=config.n_layer,
                norm=norm(config.dim_emb, eps=1e-5) if config.norm_first else None,
                config=config,
            )

        self.seq_len_per_step = config.embedder.patch_num**2+3*self.per_refine  
        mask = block_lower_triangular_mask(self.seq_len_per_step, max_data_len, use_float=True)
        self.register_buffer("mask", mask, persistent=False)  

        if self.flex_attn:
            block_size = config.patch_num**2+3*self.per_refine 
            seq_len = block_size * (max_data_len - 1)
            self.block_mask = create_block_mask(
                partial(block_causal, block_size=block_size), None, None, seq_len, seq_len
            )
            self.block_size = block_size
            self.block_mask_prefil = None


    def summary(self): 
        s = "\n"
        s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        s += f"\tTransformer:    {sum([p.numel() for p in self.transformer.parameters() if p.requires_grad]):,}"
        return s

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd": 
            return self.fwd(**kwargs)
        elif mode == "generate": 
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data, times, input_len: int, **kwargs):
        """
        Inputs:
            data:          Tensor     (bs, input_len+output_len, x_num, x_num, data_dim)
            times:         Tensor     (bs/1, input_len+output_len, 1)
            input_len:     How many timesteps to use as input, for training this should be 1

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        bs=data.size(0)
        data = data[:, :-1]  # ignore last timestep for autoregressive training (b, t_num-1, x_num, x_num, data_dim)
        times = times[:, :-1]  # (bs/1, t_num-1, 1)

        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data_input (bs, t_num-1, x_num, x_num, data_dim) -> (bs, data_len, dim)
                       data_len = (input_len + output_len - 1) * patch_num * patch_num
        """

        data,idx,refine_idx,grid_all,depth = self.embedder.encode(data, times)  # (bs, data_len, dim)
        """
        Step 2: Transformer
            data_input:   Tensor     (bs, data_len, dim)
        """
        data_len = data.size(1)
        if self.flex_attn:
            block_mask = self.block_mask
            data_encoded = self.transformer(data, block_mask=block_mask)  # (bs, data_len, dim)
        else:

            mask = self.mask[:data_len, :data_len]

            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                data_encoded = self.transformer(data,grid_all,depth,seq_len_per_step=self.seq_len_per_step, mask=mask)  # (bs, data_len, dim)

        """
        Step 3: Decode data
        """
        input_seq_len = (input_len - 1) * self.seq_len_per_step
        data_output = data_encoded[:, input_seq_len:]  # (bs, output_len*patch_num*patch_num, dim)

        idx=rearrange(idx, "(b t) s -> b t s",b=bs)
        idx=idx[:,input_len - 1:]
        idx=rearrange(idx, "b t s -> (b t) s")

        refine_idx=rearrange(refine_idx, "(b t) s -> b t s",b=bs)
        refine_idx=refine_idx[:,input_len - 1:]
        refine_idx=rearrange(refine_idx, "b t s -> (b t) s")

        data_output = self.embedder.decode(data_output,idx,refine_idx)  # (bs, output_len, x_num, x_num, data_dim)
        return data_output

    @torch.compiler.disable()
    def generate(self, data_input, times, input_len: int, data_mask, carry_over_c=-1, **kwargs):

        t_num = times.size(1) 
        output_len = t_num - input_len 
        bs, _, x_num, _, data_dim = data_input.size()

        data_all = torch.zeros(bs, t_num, x_num, x_num, data_dim, dtype=data_input.dtype, device=data_input.device)
        data_all[:, :input_len] = data_input 
        cur_len = input_len  
        prev_len = 0 

        config = self.config
        if config.kv_cache:  
            cache = KVCache(
                config.n_layer, data_input.shape[0], self.mask.size(0), config.n_head, config.dim_emb // config.n_head
            )

        if self.flex_attn and self.block_mask_prefil is None:
            seq_len_eval = self.block_size * input_len
            self.block_mask_prefil = create_block_mask(
                partial(block_causal, block_size=self.block_size), None, None, seq_len_eval, seq_len_eval
            )

        for i in range(output_len):
            cur_data_input = data_all[:, :cur_len]  # (bs, cur_len, x_num, x_num, data_dim)

            # (bs, cur_len, x_num, x_num, data_dim) -> (bs, data_len=cur_len*p*p, dim)
            skip_len = prev_len if self.config.kv_cache else 0
            cur_data_input,idx,refine_idx,grid_all,depth = self.embedder.encode(   
                cur_data_input, times[:, :cur_len], skip_len=skip_len
            )  # (bs, data_len, dim)

            mask = block_mask = None
            if (not self.config.kv_cache) or i == 0:
                if self.flex_attn:
                    block_mask = self.block_mask_prefil
                else:
                    data_len = cur_len * self.seq_len_per_step
                    mask = self.mask[:data_len, :data_len]

            if self.config.kv_cache:
                cur_data_encoded = self.transformer(cur_data_input,grid_all,depth,self.seq_len_per_step, mask, block_mask=block_mask, cache=cache)
            else:
                cur_data_encoded = self.transformer(cur_data_input,grid_all,depth,self.seq_len_per_step, mask, block_mask=block_mask)  # (bs, data_len, dim)

            new_output = cur_data_encoded[:, -self.seq_len_per_step :]  # (bs, patch_num*patch_num, dim)

            idx=rearrange(idx, "(b t) s -> b t s",b=bs)
            idx=idx[:,-1:,:]
            idx=rearrange(idx, "b t s -> (b t) s")

            refine_idx=rearrange(refine_idx, "(b t) s -> b t s",b=bs)
            refine_idx=refine_idx[:,-1:,:]
            refine_idx=rearrange(refine_idx, "b t s -> (b t) s")

            new_output = self.embedder.decode(new_output,idx,refine_idx)  

            new_output = new_output * data_mask  

            if carry_over_c >= 0:  
                new_output[:, 0, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

            data_all[:, cur_len : cur_len + 1] = new_output
            prev_len = cur_len
            cur_len += 1

        return data_all[:, input_len:]
