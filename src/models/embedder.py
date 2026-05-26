import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from neuralop.models import FNO2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from .attention_utils import get_embeddings, get_activation
except:
    from attention_utils import get_embeddings, get_activation
from logging import getLogger

logger = getLogger()


def get_embedder(config, x_num, max_output_dim,k):
    match config.type:
        case "linear":
            embedder = LinearEmbedder
        case "conv":
            embedder = ConvEmbedder
        case "patch":
            embedder = PatchEmbedder
        case _:
            raise ValueError(f"Unknown embedder type: {config.type}")

    return embedder(config, x_num, max_output_dim,k)


def layer_initialize(layer, mode="zero", gamma=0.01):
    # re-initialize given layer to have small outputs
    if mode == "zero":
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif mode == "uniform":
        nn.init.uniform_(layer.weight, -gamma, gamma)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, -gamma, gamma)
    else:
        raise ValueError(f"Unknown mode {mode}")
    

def expand_tensor(x, patch):
    batch, seq = x.shape
    i = x // patch  # [batch, seq]
    j = x % patch   # [batch, seq]
    
    y1 = 4 * i * patch + 2 * j            # [batch, seq]
    y2 = 4 * i * patch + 2 * j + 1        # [batch, seq]
    y3 = (4 * i + 2) * patch + 2 * j      # [batch, seq]
    y4 = (4 * i + 2) * patch + 2 * j + 1  # [batch, seq]

    y = torch.stack([y1, y2, y3, y4], dim=2)  # [batch, seq, 4]
    y = y.view(batch, 4 * seq)
    
    return y

def reshape_tensor(input_tensor, h, w, patch_size): 
    """
    Args:
        input_tensor: [batch, seq], where seq = h * w
        h: height of the thinginal grid
        w: width of the thinginal grid
        patch_size: size of each patch (patch = square, patch_size x patch_size)
    Returns:
        output_tensor: [batch, h*patch_size, w*patch_size]
    """
    batch_size = input_tensor.shape[0]
    
    reshaped = input_tensor.view(batch_size, h, w)

    # Unsqueeze to [batch, h, w, 1, 1] and then expand
    expanded = reshaped.unsqueeze(-1).unsqueeze(-1)  # [batch, h, w, 1, 1]
    expanded = expanded.expand(-1, -1, -1, patch_size, patch_size)  # [batch, h, w, patch, patch]
    
    output = expanded.permute(0, 1, 3, 2, 4).contiguous()  # [batch, h, patch, w, patch]
    output = output.view(batch_size, h * patch_size, w * patch_size)
    
    return output


def replace_tokens_with_groups(a, b, idxs):
    B, S, D = a.shape
    K = idxs.shape[1]

    device = a.device
    dtype = a.dtype
    idxs = idxs.long()

    is_replaced = torch.zeros((B, S), dtype=torch.long, device=device)
    is_replaced.scatter_(1, idxs, 1)

    prefix_before = torch.cumsum(is_replaced, dim=1) - is_replaced  # [B, S]
    pos_base = torch.arange(S, device=device).unsqueeze(0).expand(B, S) + 3 * prefix_before  # [B, S]

    L = S + 3 * K
    out = torch.zeros((B, L, D), dtype=dtype, device=device)  # [B, L, D]

    depth = torch.zeros((B, L), dtype=torch.long, device=device)  # [B, L]


    mask_not = (is_replaced == 0)  # [B, S]
    src_orig = a * mask_not.unsqueeze(-1)  
    index_orig = pos_base.unsqueeze(-1).expand(B, S, D)  # [B, S, D]
    out.scatter_(1, index_orig, src_orig)


    base_for_idxs = pos_base.gather(1, idxs)  # [B, K]
    offsets = torch.arange(4, device=device).view(1, 1, 4)  # [1,1,4]
    pos_mat = base_for_idxs.unsqueeze(2) + offsets  # [B, K, 4]


    pos_b_flat = pos_mat.contiguous().view(B, 4 * K)


    index_b = pos_b_flat.unsqueeze(-1).expand(B, 4 * K, D)  # [B, 4K, D]
    out.scatter_(1, index_b, b)
    depth.scatter_(1, pos_b_flat, torch.ones_like(pos_b_flat, dtype=depth.dtype, device=device))

    return out, depth



def load_ckpt_into_model(model, ckpt_path: str, device: str = "cpu", strict: bool = True):
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)



def freeze_model(model: nn.Module, train_bn: bool = False):
    for p in model.parameters():
        p.requires_grad_(False)

    if not train_bn:
        model.eval()
    return model






def build_grid_coords(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        ys = (torch.arange(H, device=device) + 0.5) / H  # [H]
        xs = (torch.arange(W, device=device) + 0.5) / W  # [W]
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")   # [H, W]

        coords_hw = torch.stack([xx, yy], dim=-1)        # [H, W, 2]
        coords = coords_hw.view(1, H * W, 2).expand(B, H * W, 2).contiguous()
        return coords                                    # [B, H*W, 2]

def add_pos_and_replace(a, b, idxs, pos): 
    B, S, D = a.shape
    K = idxs.shape[1]
    device = a.device


    if pos.shape[0] == 1:
        pos_a = pos.expand(B, S, D)
    elif pos.shape[0] == B:
        pos_a = pos


    pos_for_idxs = pos_a.gather(1, idxs.unsqueeze(-1).expand(B, K, D))  # [B, K, D]
    pos_b = pos_for_idxs.repeat_interleave(4, dim=1)  # [B, g*K, D]   
    a_pos = a + pos_a
    b_pos = b + pos_b
    out,depth = replace_tokens_with_groups(a_pos, b_pos, idxs)
    return out


def split_merged(merged: torch.Tensor, idxs: torch.Tensor, seq_len: int, group_size: int = 4):
    B, L, D = merged.shape
    S = seq_len
    K = idxs.shape[1]
    device = merged.device

    is_replaced = torch.zeros((B, S), dtype=torch.long, device=device)  # long !!
    is_replaced.scatter_(1, idxs, torch.ones_like(idxs, dtype=is_replaced.dtype))
    prefix_before = torch.cumsum(is_replaced, dim=1) - is_replaced  # [B, S], long
    pos_base = torch.arange(S, device=device).unsqueeze(0).expand(B, S) + (group_size - 1) * prefix_before

    idx_expand = pos_base.unsqueeze(-1).expand(B, S, D)  # [B, S, D]
    gathered = merged.gather(1, idx_expand)               # [B, S, D]

    mask_not = (is_replaced == 0)                         # [B, S], bool
    a_recovered = torch.where(mask_not.unsqueeze(-1), gathered, torch.zeros_like(gathered))

    base_for_idxs = pos_base.gather(1, idxs)              # [B, K]
    offsets = torch.arange(group_size, device=device).view(1, 1, group_size)  # [1,1,g]
    pos_mat = base_for_idxs.unsqueeze(2) + offsets        # [B, K, g]
    pos_flat = pos_mat.contiguous().view(B, group_size * K)  # [B, g*K]
    idx_expand_b = pos_flat.unsqueeze(-1).expand(B, group_size * K, D)        # [B, gK, D]
    b_recovered = merged.gather(1, idx_expand_b)          # [B, gK, D]

    return a_recovered, b_recovered


def scatter_to_length(x: torch.Tensor, idx: torch.Tensor, L: int) -> torch.Tensor:
    if idx.dtype != torch.long:
        idx = idx.long()

    B, N, D = x.shape
    device = x.device
    dtype = x.dtype

    if torch.any(idx < 0) or torch.any(idx >= L):
        raise IndexError(f"idx contains out-of-range values (should be in [0, {L-1}])")

    out = torch.zeros((B, L, D), dtype=dtype, device=device)

    idx_exp = idx.unsqueeze(-1).expand(B, N, D)  # [B, N, D]

    out.scatter_(1, idx_exp, x)

    return out

def straight_through_topk(probs: torch.Tensor, k: int, dim: int = -1, eps: float = 1e-12):
    # hard one-hot（multi-hot）
    _, idx = probs.topk(k, dim=dim)
    hard = torch.zeros_like(probs).scatter_(dim, idx, 1.0)

    y = hard + (probs - probs.detach())
    return idx,y

def gumbel_noise(shape, device, eps=1e-10):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def straight_through_gumbel_topk(
    probs: torch.Tensor, 
    k: int, 
    dim: int = -1, 
    temperature: float = 1.0,
    noise_scale: float = 1.0,
    dropout_p: float = 0.0,
    eps: float = 1e-12
):
    # 1. Dropout exploration
    if dropout_p > 0 and probs.requires_grad:
        probs = F.dropout(probs, p=dropout_p, training=True)

    gumbel = noise_scale * gumbel_noise(probs.size(), probs.device)
    noisy_logits = torch.log(probs + eps) + gumbel

    soft = F.softmax(noisy_logits / temperature, dim=dim)

    # 4. Hard Top-k
    _, idx = noisy_logits.topk(k, dim=dim)
    hard = torch.zeros_like(probs).scatter_(dim, idx, 1.0)

    # 5. ST trick
    y = hard + (soft - soft.detach())

    return idx, y

def masked_index(tensor, mask):
    """
    Args:
        tensor: shape [batch, seq, dim]
        mask: shape [batch, seq], containing k 1s per batch, rest 0s
    Returns:
        result: shape [batch, k, dim]
    """
    # Expand mask to [batch, seq, dim]
    mask_expanded = mask.unsqueeze(-1).expand_as(tensor)
    
    # Apply mask to zero out unwanted elements
    masked_tensor = tensor * mask_expanded
    
    # Reshape to [batch, seq * dim]
    batch_size, seq_len, dim = tensor.shape
    masked_tensor_flat = masked_tensor.view(batch_size, -1)
    
    # Create a mask for non-zero elements [batch, seq * dim]
    non_zero_mask = (masked_tensor_flat != 0)
    
    # For each batch, select non-zero elements and reshape to [k, dim]
    result = []
    for i in range(batch_size):
        non_zero_elements = masked_tensor_flat[i][non_zero_mask[i]]
        result.append(non_zero_elements.view(-1, dim))
    
    # Stack all batches to get [batch, k, dim]
    return torch.stack(result, dim=0)





def hardest_patches(data: torch.Tensor, patch_size: int, k: int):
    assert data.ndim == 4, "Expect data shape [B,C,H,W]"
    B, C, H, W = data.shape
    device = data.device
    ph, pw = patch_size, patch_size
    assert ph > 0 and pw > 0

    pad_h = (ph - (H % ph)) % ph
    pad_w = (pw - (W % pw)) % pw
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    if pad_h != 0 or pad_w != 0:
        data = F.pad(data, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        H += pad_h
        W += pad_w

    sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=device).view(1,1,3,3).repeat(C,1,1,1)
    sobel_y = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], device=device).view(1,1,3,3).repeat(C,1,1,1)
    gx = F.conv2d(data, sobel_x, groups=C, padding=1)
    gy = F.conv2d(data, sobel_y, groups=C, padding=1)
    grad_mag = torch.sqrt(gx * gx + gy * gy).mean(dim=1, keepdim=True)  # [B,1,H,W]

    lap = torch.tensor([[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]], device=device).view(1,1,3,3).repeat(C,1,1,1)
    hf_map = F.conv2d(data, lap, groups=C, padding=1).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]

    local_mean = F.avg_pool2d(data, kernel_size=(ph, pw))  # [B,C,H//ph,W//pw]
    global_mean = data.mean(dim=[2, 3], keepdim=True)       # [B,C,1,1]
    midanom = (local_mean - global_mean).abs().mean(dim=1)  # [B,H//ph,W//pw]

    grad_score = F.avg_pool2d(grad_mag, kernel_size=(ph, pw)).squeeze(1)  # [B,H_p,W_p]
    hf_score = F.avg_pool2d(hf_map, kernel_size=(ph, pw)).squeeze(1)      # [B,H_p,W_p]

    def norm(x): return (x - x.amin(dim=(-2, -1), keepdim=True)) / (x.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    score = norm(grad_score) + norm(hf_score) + norm(midanom)  # [B,H_p,W_p]

    B, nH, nW = score.shape
    num_patches = nH * nW
    flat = score.view(B, -1)
    k_clamped = min(max(0, int(k)), num_patches)

    if k_clamped == 0:
        ids = torch.zeros((B, 0), dtype=torch.long, device=device)
        onehot = torch.zeros((B, num_patches), dtype=torch.long, device=device)
        return ids, onehot

    topk_vals, ids = torch.topk(flat, k_clamped, dim=1)  # [B, k]
    onehot = torch.zeros((B, num_patches), dtype=torch.long, device=device)
    onehot.scatter_(1, ids, 1)

    return ids, onehot


class SoftTopKCodebookDecoder(nn.Module):

    def __init__(self, channel,dim,patch, k=4,  temperature=1.0,training=1): 
        super().__init__()
        self.dim=dim
        self.k = k
        self.codebook_size = patch*patch
        self.temperature = temperature
        self.conv = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(256, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1)  
        )
        self.projection = nn.Linear(dim, self.codebook_size)
        self.training=training

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        return: soft/hard (ST) multi-hot mask, shape [batch, codebook_size]
        """
        x=self.conv(x).view(x.size(0), -1)


        logits = self.projection(x)     # [B, C]
        probs = F.softmax(logits / self.temperature, dim=-1)  # [B, C]

        if self.training:
            idx,soft_topk_mask = straight_through_topk(probs, k=self.k, dim=-1)
        else:
            _, idx = probs.topk(self.k, dim=-1)
            hard = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
            soft_topk_mask=hard
        return torch.sort(idx, dim=1)[0],soft_topk_mask  


class SinusoidalPositionalEncoding3D(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEncoding3D, self).__init__()
        self.d_model = d_model

        self.d_model_x = d_model // 3
        self.d_model_y = d_model // 3
        self.d_model_depth = d_model - self.d_model_x - self.d_model_y  

    def create_position_encoding(self, pos, d_model):
        pe = torch.zeros(pos.size(0), d_model).to(pos.device)  # (b*n, d_model)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(pos.device)

        pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div_term)
        else:
            pe[:, 1:d_model:2] = torch.cos(pos.unsqueeze(1) * div_term[:-1])

        return pe

    def forward(self, x, y, depth):
        batch_size, n = x.size()

        x_flat = x.view(-1)  # (b*n,)
        y_flat = y.view(-1)  # (b*n,)
        depth_flat = depth.view(-1)  # (b*n,)

        pe_x = self.create_position_encoding(x_flat, self.d_model_x)  # (b*n, d_model_x)
        pe_y = self.create_position_encoding(y_flat, self.d_model_y)  # (b*n, d_model_y)
        pe_depth = self.create_position_encoding(depth_flat, self.d_model_depth)  # (b*n, d_model_depth)
        pos_encoding = torch.cat([pe_x, pe_y, pe_depth], dim=1)  # (b*n, d_model)
        pos_encoding = pos_encoding.view(batch_size, n, self.d_model)

        return pos_encoding



def random_k_mask(batch, seq, k, device=None):
    rand_perm = torch.rand(seq, batch, device=device).argsort(dim=0).T  # [batch, seq]
    idx = rand_perm[:, :k]  # [batch, k]
    mask = torch.zeros(batch, seq, dtype=torch.float32, device=device)
    mask.scatter_(1, idx, 1.0) 
    
    return idx, mask


    

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "gn", groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)

        if norm == "bn":
            self.n1 = nn.BatchNorm2d(out_ch)
            self.n2 = nn.BatchNorm2d(out_ch)
        elif norm == "in":
            self.n1 = nn.InstanceNorm2d(out_ch, affine=True)
            self.n2 = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm == "gn":
            g1 = min(groups, out_ch)
            g2 = min(groups, out_ch)
            self.n1 = nn.GroupNorm(g1, out_ch)
            self.n2 = nn.GroupNorm(g2, out_ch)
        else:
            raise ValueError(f"Unknown norm={norm}")

        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.n1(self.conv1(x)))
        x = self.act(self.n2(self.conv2(x)))
        return x


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "gn"):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = ConvBlock(in_ch, out_ch, norm=norm)

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str = "gn", up_mode: str = "bilinear"):
        super().__init__()
        self.up_mode = up_mode
        if up_mode == "deconv":
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        elif up_mode == "bilinear":
            self.up = None
        else:
            raise ValueError(f"Unknown up_mode={up_mode}")

        self.block = ConvBlock(in_ch + skip_ch, out_ch, norm=norm)

    def forward(self, x, skip):
        if self.up_mode == "deconv":
            x = self.up(x)
        else:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        # 对齐（极少数 odd 尺寸会出现 1 像素偏差）
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return x


# -----------------------------
# UNet-based refinement network
# -----------------------------
class UNetRefineNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base: int = 32,
        patch_size: int = 8,
        depth: int = 3,
        norm: str = "gn",
        up_mode: str = "bilinear",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.depth = depth

        # encoder
        self.inc = ConvBlock(in_channels, base, norm=norm)

        chs = [base]
        downs = []
        for i in range(depth):
            in_ch = chs[-1]
            out_ch = base * (2 ** (i + 1))
            downs.append(Down(in_ch, out_ch, norm=norm))
            chs.append(out_ch)
        self.downs = nn.ModuleList(downs)

        # bottleneck
        self.bottleneck = ConvBlock(chs[-1], chs[-1], norm=norm)

        # decoder
        ups = []
        for i in reversed(range(depth)):
            in_ch = chs[i + 1]
            skip_ch = chs[i]
            out_ch = chs[i]
            ups.append(Up(in_ch=in_ch, skip_ch=skip_ch, out_ch=out_ch, norm=norm, up_mode=up_mode))
        self.ups = nn.ModuleList(ups)

        self.head = nn.Sequential(
            nn.Conv2d(base, base, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(base, 1, kernel_size=1),
        )

    def forward(self, x):
        BT, C, H, W, = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"H,W must be divisible by patch_size={p}"



        # encoder with skips
        skips = []
        x = self.inc(x)         # [BT,base,H,W]
        skips.append(x)

        for down in self.downs:
            x = down(x)         # spatial /2 each stage
            skips.append(x)

        # bottleneck
        x = self.bottleneck(x)

        # decoder (reverse skips; last skip corresponds to deepest feature, so pop in reverse)
        # skips list: [lvl0, lvl1, ..., lvl_depth]
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            x = up(x, skip)

        # patch pooling: [BT,base,H,W] -> [BT,base,Ph,Pw]
        Ph, Pw = H // p, W // p
        x = x.view(BT, x.size(1), Ph, p, Pw, p).mean(dim=(3, 5))

        # head: [BT,1,Ph,Pw] -> [BT,Ph*Pw]
        logits = self.head(x).squeeze(1)          # [BT,Ph,Pw]
        logits_flat = logits.flatten(1)           # [BT,P]
        return logits_flat


@torch.no_grad()
def logits_to_random_k_idx(logits_flat: torch.Tensor, k: int, device=None, largest: bool = True):
    assert logits_flat.dim() == 2
    N, S = logits_flat.shape
    k = min(k, S)
    if device is None:
        device = logits_flat.device
    idx = torch.topk(logits_flat, k=k, dim=1, largest=largest, sorted=False).indices  # [N,k]

    mask = torch.zeros(N, S, dtype=torch.float32, device=device)
    mask.scatter_(1, idx, 1.0)
    return idx, mask





class LinearEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim
        act = get_activation("gelu")

        assert (
            x_num % config.patch_num == 0
        ), f"x_num must be divisible by patch_num, x_num: {x_num}, patch_num: {config.patch_num}"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        fold_params = dict(kernel_size=self.patch_resolution, stride=self.patch_resolution)
        self.patchify = nn.Unfold(**fold_params)
        self.unpatchify = nn.Fold(output_size=x_num, **fold_params)

        self.patch_position_embeddings = get_embeddings((1, 1, config.patch_num * config.patch_num, self.dim))
        self.time_embeddings = get_embeddings((1, config.get("max_time_len", 20), 1, self.dim))

        self.in_proj = nn.Linear(self.patch_dim, self.dim)
        self.pre_proj = nn.Sequential(
            act(),
            nn.Linear(self.dim, self.dim),
        )

        self.conv_dim = config.get("conv_dim", self.dim // 4)
        self.post_proj = nn.Sequential(
            nn.Linear(self.dim, self.conv_dim),
            act(),
            nn.Linear(self.conv_dim, self.conv_dim),
            act(),
        )
        self.head = nn.Linear(self.conv_dim, self.patch_dim)

    def get_pos_embeddings(self, t_len):
        return (self.time_embeddings[:, :t_len] + self.patch_position_embeddings).view(1, -1, self.dim)  # (1, t*p*p, d)

    def encode(self, data, proj=True):
        """
        (b, t, x_num, x_num, data_dim) -> (b, t*p*p, d)
        """
        b = data.size(0)
        data = rearrange(data, "b t h w c -> (b t) c h w")
        data = self.patchify(data)  # (b*t, d, p*p)
        data = rearrange(data, "(b t) d pp -> b (t pp) d", b=b)

        if proj:
            return self.pre_proj(self.in_proj(data))
        else:
            return data

    def decode(self, data, proj=True):
        """
        (b, t*p*p, d) -> (b, t, x_num, x_num, data_dim)
        """
        if proj:
            data = self.head(self.post_proj(data))  # (b, t*p*p, d)

        b = data.size(0)
        data = rearrange(data, "b (t pp) d -> (b t) d pp", pp=self.config.patch_num**2)
        data = self.unpatchify(data)  # (b*t, data_dim, x_num, x_num)
        data = rearrange(data, "(b t) c h w -> b t h w c", b=b)

        return data


class ConvEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim,k,temperature=1.0,training=1):
        super().__init__()
        self.config = config
        self.topk=k
        self.dim = config.dim
        self.data_dim = data_dim
        act = get_activation("gelu")

        assert (
            x_num % config.patch_num == 0
        ), f"x_num must be divisible by patch_num, x_num: {x_num}, patch_num: {config.patch_num}"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        assert (
            x_num % config.patch_num_output == 0
        ), f"x_num must be divisible by patch_num_output, x_num: {x_num}, patch_num_output: {config.patch_num_output}"
        self.patch_resolution_output = (
            x_num // config.patch_num_output
        )  # resolution of one space dimension for each patch in output
        self.patch_dim_output = (
            data_dim * self.patch_resolution_output * self.patch_resolution_output
        )  # dimension per patch in output

        ## for encoder part
        if config.select=='model':
            self.select=SoftTopKCodebookDecoder(self.data_dim,self.dim, config.patch_num, k=self.topk)
        if config.select=='compare':
            self.select=UNetRefineNet(in_channels=self.data_dim,patch_size=x_num//config.patch_num)
            CKPT_PATH = ""
            load_ckpt_into_model(self.select, CKPT_PATH, strict=True)
            self.select=freeze_model(self.select)

        self.patch_position_embeddings = get_embeddings((1, config.patch_num * config.patch_num, self.dim)) 
        self.patch_position_embeddings_sub = get_embeddings((1, 1, 2 * 2, self.dim)) 

        self.positional_encoding_3d = SinusoidalPositionalEncoding3D(self.dim)

        self.time_embed_type = config.get("time_embed", "continuous") 
        match self.time_embed_type:
            case "continuous":
                self.time_proj = nn.Sequential(
                    nn.Linear(1, self.dim),
                    act(),
                    nn.Linear(self.dim, self.dim),
                )
            case "learnable":
                self.time_embeddings = get_embeddings((1, config.get("max_time_len", 10), 1, self.dim))

        if config.get("early_conv", 0): 
            n_conv_layers = math.log2(self.patch_resolution) 
            assert n_conv_layers.is_integer(), f"patch_resolution {self.patch_resolution} must be a power of 2"
            n_conv_layers = int(n_conv_layers)
            kernel_size = [3] * n_conv_layers + [1] 
            stride = [2] * n_conv_layers + [1] 
            padding = [1] * n_conv_layers + [0] 
            channels = [data_dim] + [self.dim // (2**i) for i in range(n_conv_layers - 1, 0, -1)] + [self.dim, self.dim] 
            self.conv_proj = nn.Sequential()
            for i in range(len(kernel_size)):
                self.conv_proj.append(
                    nn.Conv2d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_size[i],
                        stride=stride[i],
                        padding=padding[i],
                    )
                )
                if i < len(kernel_size) - 1:
                    self.conv_proj.append(act())
        else: 
            self.in_proj = nn.Conv2d(
                in_channels=data_dim,
                out_channels=self.dim,
                kernel_size=self.patch_resolution,
                stride=self.patch_resolution,
                bias=False
            )
            self.conv_proj = nn.Sequential(
                act(),
                nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1,bias=False),
            )

            self.in_proj_sub = nn.Conv2d(  
                in_channels=data_dim,
                out_channels=self.dim,
                kernel_size=self.patch_resolution//2,
                stride=self.patch_resolution//2,
                bias=False,
            )
            self.conv_proj_sub = nn.Sequential(
                act(),
                nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1,bias=False,),
            )

        ## for decoder part

        self.conv_dim = config.get("conv_dim", self.dim // 4)

        if config.get("deep", 0): 
            self.post_proj = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=self.dim),
                act(),
                nn.Linear(in_features=self.dim, out_features=self.dim),
                act(),
                Rearrange("b (t h w) d -> (b t) d h w", h=self.config.patch_num_output, w=self.config.patch_num_output),
                nn.ConvTranspose2d(
                    in_channels=self.dim,
                    out_channels=self.conv_dim,
                    kernel_size=self.patch_resolution_output,
                    stride=self.patch_resolution_output,
                ),
                act(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                act(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                act(),
            )
        else:
            self.post_proj = nn.Sequential(
                Rearrange("b (h w) d -> b d h w", h=self.config.patch_num, w=self.config.patch_num),
                nn.ConvTranspose2d(
                    in_channels=self.dim,
                    out_channels=self.conv_dim,
                    kernel_size=self.patch_resolution_output,
                    stride=self.patch_resolution_output,
                    bias=False
                ),
            )
            self.post_proj_sub = nn.Sequential(
                    Rearrange("b (h w) d -> b d h w", h=self.config.patch_num_output*2, w=self.config.patch_num_output*2),
                    nn.ConvTranspose2d(
                        in_channels=self.dim,
                        out_channels=self.conv_dim,
                        kernel_size=self.patch_resolution_output//2,
                        stride=self.patch_resolution_output//2,
                        bias=False,
                    ),
                )
        self.head = nn.Sequential(
                act(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                act(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.data_dim, kernel_size=1, stride=1)
        )
        

        if config.get("initialize_small_output", 0):
            layer_initialize(self.head, mode=config.initialize_small_output)

    def encode(self, data, times,skip_len=0): 
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs, input_len, 1)
        Output:
            data:           Tensor (bs, data_len, dim)      data_len = input_len * patch_num * patch_num
                            embedded data + time embeddings + patch position embeddings
        """

        bs = data.size(0) 
        data = rearrange(data[:, skip_len:], "b t h w c -> (b t) c h w")
        idx_bs=data.size(0)
        if self.config.select=='rand':
            idx,indexs=random_k_mask(idx_bs, self.config.patch_num**2, self.topk, device=data.device)
        elif self.config.select=='physical':
            idx,indexs=hardest_patches(data,self.patch_resolution,self.topk)
        elif self.config.select=='model':
            idx,indexs=self.select(data)
            print(idx)
        elif self.config.select=='compare':
            self.select=freeze_model(self.select)
            logits=self.select(data)
            idx,indexs=logits_to_random_k_idx(logits_flat=logits,k=self.topk,device=data.device)
            

        refine_mask=reshape_tensor(indexs,self.config.patch_num,self.config.patch_num,self.patch_resolution).unsqueeze(1)

        data_ori=data*(1-refine_mask)
        data_thin=data*refine_mask

        data_ori = self.in_proj(data_ori)
        data_ori = self.conv_proj(data_ori) 
        data_ori = rearrange(data_ori, "b d h w -> b (h w) d") 
        data_thin = self.in_proj_sub(data_thin)
        data_thin = self.conv_proj_sub(data_thin) 
        data_thin = rearrange(data_thin, "b d h w -> b (h w) d")  

        refine_idx=expand_tensor(idx,self.config.patch_num)  

        data_thin = torch.gather(data_thin, dim=1, index=refine_idx.unsqueeze(-1).expand(-1, -1, self.dim))
        data,depth=replace_tokens_with_groups(data_ori, data_thin, idx)

        grid_ori=build_grid_coords(data.shape[0],self.config.patch_num,self.config.patch_num,data_ori.device)
        grid_thin=build_grid_coords(data.shape[0],2*self.config.patch_num,2*self.config.patch_num,data_thin.device)

        grid_thin = torch.gather(grid_thin, dim=1, index=refine_idx.unsqueeze(-1).expand(-1, -1, 2))
        grid_all,depth=replace_tokens_with_groups(grid_ori, grid_thin, idx)
        
        data=rearrange(data, "(b t) s d -> b t s d",b=bs) 



        match self.time_embed_type:
            case "continuous":
                time_embeddings = self.time_proj(times[:, skip_len:])[:, :, None]  # (bs, input_len, 1, dim)
                data = data + time_embeddings
            case "learnable":
                time_embeddings = self.time_embeddings[:, skip_len : times.size(1)]  # (bs, input_len, 1, dim)
                data = data + time_embeddings


        data = data.reshape(bs, -1, self.dim)

        return data,idx,refine_idx,grid_all,depth

    def decode(self, data_output,idx,refine_idx):
        """
        Input:
            data_output:     Tensor (bs, query_len, dim)
                             query_len = output_len * patch_num * patch_num
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)
        data=rearrange(data_output,"b (t s) d -> (b t) s d", s=self.config.patch_num*self.config.patch_num+3*self.topk)
        data_ori,data_thin=split_merged(data, idx, self.config.patch_num*self.config.patch_num)
       
        data_ori = self.post_proj(data_ori)  # (bs*output_len, data_dim, x_num, x_num)


        data_thin=scatter_to_length(data_thin,refine_idx,self.config.patch_num*self.config.patch_num*2*2)

        data_thin = self.post_proj_sub(data_thin)

        data_output=data_ori+data_thin
        data_output = self.head(data_output)
        data_output = rearrange(data_output, "(b t) c h w -> b t h w c", b=bs)
        return data_output


class PatchEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim
        act = get_activation("gelu")

        assert (
            x_num % config.patch_num == 0
        ), f"x_num must be divisible by patch_num, x_num: {x_num}, patch_num: {config.patch_num}"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        assert (
            x_num % config.patch_num_output == 0
        ), f"x_num must be divisible by patch_num_output, x_num: {x_num}, patch_num_output: {config.patch_num_output}"
        self.patch_resolution_output = (
            x_num // config.patch_num_output
        )  # resolution of one space dimension for each patch in output
        self.patch_dim_output = (
            data_dim * self.patch_resolution_output * self.patch_resolution_output
        )  # dimension per patch in output

        ## for encoder part

        self.patch_position_embeddings = get_embeddings((1, 1, config.patch_num, config.patch_num, self.dim))

        self.time_embed_type = config.get("time_embed", "continuous")
        match self.time_embed_type:
            case "continuous":
                self.time_proj = nn.Sequential(
                    nn.Linear(1, self.dim),
                    act(),
                    nn.Linear(self.dim, self.dim),
                )
            case "learnable":
                self.time_embeddings = get_embeddings((1, config.get("max_time_len", 20), 1, 1, self.dim))

        # regular vit patch embedding
        self.in_proj = nn.Conv2d(
            in_channels=data_dim,
            out_channels=self.dim,
            kernel_size=self.patch_resolution,
            stride=self.patch_resolution,
        )
        self.conv_proj = nn.Sequential(
            act(),
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1),
        )

        ## for decoder part

        self.conv_dim = config.get("conv_dim", self.dim // 4)

        self.post_proj = nn.Sequential(
            # Rearrange("b (t h w) d -> (b t) d h w", h=self.config.patch_num_output, w=self.config.patch_num_output),
            nn.ConvTranspose2d(
                in_channels=self.dim,
                out_channels=self.conv_dim,
                kernel_size=self.patch_resolution_output,
                stride=self.patch_resolution_output,
            ),
            act(),
            nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
            act(),
        )
        self.head = nn.Conv2d(in_channels=self.conv_dim, out_channels=self.data_dim, kernel_size=1, stride=1)

    def encode(self, data, times, mode="none"):
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs, input_len, 1)
        Output:
            data:   embedded data + time embeddings + patch position embeddings
                mode:   flatten -> Tensor (bs, input_len*patch_num*patch_num, dim)
                        st      -> Tensor (bs, input_len, patch_num*patch_num, dim)
                        none    -> Tensor (bs, input_len, patch_num, patch_num, dim)
        """

        bs = data.size(0)
        data = rearrange(data, "b t h w c -> (b t) c h w")
        data = self.in_proj(data)
        data = self.conv_proj(data)  # (bs*input_len, d, patch_num, patch_num)
        data = rearrange(data, "(b t) d h w -> b t h w d", b=bs)  # (bs, input_len, p, p, dim)

        match self.time_embed_type:
            case "continuous":
                time_embeddings = self.time_proj(times)[:, :, None, None]  # (bs, input_len, 1, 1, dim)
                data = data + time_embeddings
            case "learnable":
                time_embeddings = self.time_embeddings[:, : times.size(1)]  # (bs, input_len, 1, 1, dim)
                data = data + time_embeddings

        data = data + self.patch_position_embeddings  # (b, input_len, p*p, d)

        match mode:
            case "flatten":
                return data.reshape(bs, -1, self.dim)
            case "st":
                # space time
                return rearrange(data, "b t h w c -> b t (h w) c")
            case _:
                return data

    def decode(self, data_output, mode="none"):
        """
        Input:
            data_output:
                mode:   flatten -> Tensor (bs, output_len*patch_num*patch_num, dim)
                        st      -> Tensor (bs, output_len, patch_num*patch_num, dim)
                        none    -> Tensor (bs, output_len, patch_num, patch_num, dim)
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)

        match mode:
            case "flatten":
                data_output = rearrange(
                    data_output,
                    "b (t h w) d -> (b t) d h w",
                    h=self.config.patch_num_output,
                    w=self.config.patch_num_output,
                )
            case "st":
                data_output = rearrange(
                    data_output,
                    "b t (h w) d -> (b t) d h w",
                    h=self.config.patch_num_output,
                    w=self.config.patch_num_output,
                )
            case _:
                data_output = rearrange(data_output, "b t h w d -> (b t) d h w")

        data_output = self.post_proj(data_output)  # (bs*output_len, data_dim, x_num, x_num)
        data_output = self.head(data_output)
        data_output = rearrange(data_output, "(b t) c h w -> b t h w c", b=bs)
        return data_output
