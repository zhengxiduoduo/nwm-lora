# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import fnmatch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                              LoRA Utilities                                    #
#################################################################################

DEFAULT_LORA_TARGET_MODULES = (
    "blocks.*.attn.qkv",
    "blocks.*.attn.proj",
    "blocks.*.cttn",
    "blocks.*.mlp.fc1",
    "blocks.*.mlp.fc2",
    "blocks.*.adaLN_modulation.1",
    "final_layer.linear",
    "final_layer.adaLN_modulation.1",
)


class LoRALinear(nn.Module):
    """
    A drop-in LoRA wrapper for nn.Linear.
    The wrapped base layer is frozen; only lora_A/lora_B are trainable.
    """
    def __init__(self, base_layer, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear base layer.")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.empty(base_layer.out_features, self.rank))
        self.reset_lora_parameters()

        for param in self.base_layer.parameters():
            param.requires_grad = False

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.base_layer(x)
        lora_x = self.lora_dropout(x).to(self.lora_A.dtype)
        update = F.linear(F.linear(lora_x, self.lora_A), self.lora_B) * self.scaling
        return result + update.to(result.dtype)


class LoRAMultiheadAttention(nn.Module):
    """
    LoRA wrapper for nn.MultiheadAttention.
    It keeps the original MHA weights frozen and injects low-rank updates into
    q/k/v and output projection weights at forward time.
    """
    def __init__(self, base_layer, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base_layer, nn.MultiheadAttention):
            raise TypeError("LoRAMultiheadAttention expects an nn.MultiheadAttention base layer.")
        if not base_layer._qkv_same_embed_dim:
            raise ValueError("LoRA MHA wrapper only supports q/k/v with the same embed dim.")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        embed_dim = base_layer.embed_dim

        for name in ("q", "k", "v", "out"):
            setattr(self, f"lora_A_{name}", nn.Parameter(torch.empty(self.rank, embed_dim)))
            setattr(self, f"lora_B_{name}", nn.Parameter(torch.empty(embed_dim, self.rank)))

        self.reset_lora_parameters()
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def reset_lora_parameters(self):
        for name in ("q", "k", "v", "out"):
            nn.init.kaiming_uniform_(getattr(self, f"lora_A_{name}"), a=math.sqrt(5))
            nn.init.zeros_(getattr(self, f"lora_B_{name}"))

    def _delta(self, name, dtype):
        lora_A = getattr(self, f"lora_A_{name}")
        lora_B = getattr(self, f"lora_B_{name}")
        return (lora_B @ lora_A * self.scaling).to(dtype)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        is_batched = query.dim() == 3
        if self.base_layer.batch_first and is_batched:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        base = self.base_layer
        in_proj_delta = torch.cat(
            [
                self._delta("q", base.in_proj_weight.dtype),
                self._delta("k", base.in_proj_weight.dtype),
                self._delta("v", base.in_proj_weight.dtype),
            ],
            dim=0,
        )
        in_proj_weight = base.in_proj_weight + in_proj_delta
        out_proj_weight = base.out_proj.weight + self._delta("out", base.out_proj.weight.dtype)

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=base.embed_dim,
            num_heads=base.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=base.in_proj_bias,
            bias_k=base.bias_k,
            bias_v=base.bias_v,
            add_zero_attn=base.add_zero_attn,
            dropout_p=base.dropout,
            out_proj_weight=out_proj_weight,
            out_proj_bias=base.out_proj.bias,
            training=base.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.base_layer.batch_first and is_batched:
            attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_output_weights


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _split_target_modules(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def normalize_lora_config(config=None, args=None):
    config = config or {}
    raw_lora = config.get("lora", {})
    if isinstance(raw_lora, bool):
        raw_lora = {"enabled": raw_lora}
    lora_config = {
        "enabled": _as_bool(raw_lora.get("enabled", False)),
        "rank": int(raw_lora.get("rank", raw_lora.get("r", 8))),
        "alpha": float(raw_lora.get("alpha", 16)),
        "dropout": float(raw_lora.get("dropout", 0.0)),
        "target_modules": _split_target_modules(raw_lora.get("target_modules", DEFAULT_LORA_TARGET_MODULES)),
        "train_bias": raw_lora.get("train_bias", "none"),
    }

    if args is not None:
        if getattr(args, "lora", None) is not None:
            lora_config["enabled"] = _as_bool(args.lora)
        if getattr(args, "lora_rank", None) is not None:
            lora_config["rank"] = int(args.lora_rank)
        if getattr(args, "lora_alpha", None) is not None:
            lora_config["alpha"] = float(args.lora_alpha)
        if getattr(args, "lora_dropout", None) is not None:
            lora_config["dropout"] = float(args.lora_dropout)
        if getattr(args, "lora_target_modules", None):
            lora_config["target_modules"] = _split_target_modules(args.lora_target_modules)
        if getattr(args, "lora_train_bias", None):
            lora_config["train_bias"] = args.lora_train_bias

    return lora_config


def is_lora_enabled(lora_config):
    return bool(lora_config and _as_bool(lora_config.get("enabled", False)))


def _matches_lora_target(module_name, target_modules):
    return any(
        fnmatch.fnmatch(module_name, target)
        or module_name.endswith(target)
        or module_name == target
        for target in target_modules
    )


def apply_lora(model, lora_config):
    if not is_lora_enabled(lora_config):
        return model

    target_modules = lora_config.get("target_modules") or DEFAULT_LORA_TARGET_MODULES
    rank = int(lora_config.get("rank", 8))
    alpha = float(lora_config.get("alpha", 16))
    dropout = float(lora_config.get("dropout", 0.0))

    def _replace(module, prefix=""):
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (LoRALinear, LoRAMultiheadAttention)):
                continue
            if isinstance(child, nn.MultiheadAttention) and _matches_lora_target(full_name, target_modules):
                setattr(module, child_name, LoRAMultiheadAttention(child, rank=rank, alpha=alpha))
                continue
            if isinstance(child, nn.Linear) and _matches_lora_target(full_name, target_modules):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                continue
            _replace(child, full_name)

    _replace(model)
    mark_only_lora_as_trainable(model, train_bias=lora_config.get("train_bias", "none"))
    return model


def mark_only_lora_as_trainable(model, train_bias="none"):
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    if train_bias == "all":
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.requires_grad = True
    elif train_bias == "lora_only":
        for module in model.modules():
            if isinstance(module, LoRALinear) and module.base_layer.bias is not None:
                module.base_layer.bias.requires_grad = True
            elif isinstance(module, LoRAMultiheadAttention):
                if module.base_layer.in_proj_bias is not None:
                    module.base_layer.in_proj_bias.requires_grad = True
                if module.base_layer.out_proj.bias is not None:
                    module.base_layer.out_proj.bias.requires_grad = True
    elif train_bias != "none":
        raise ValueError("train_bias must be one of: 'none', 'all', 'lora_only'.")


def lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def _clean_state_key(key):
    key = key.replace("_orig_mod.", "")
    if key.startswith("module."):
        key = key[len("module."):]
    return key


def _find_base_layer_key(key, model_keys):
    parts = key.split(".")
    for insert_idx in range(len(parts), 0, -1):
        candidate = ".".join(parts[:insert_idx] + ["base_layer"] + parts[insert_idx:])
        if candidate in model_keys:
            return candidate
    return None


def load_lora_compatible_state_dict(model, state_dict, strict=True):
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    remapped_state = {}
    dropped_keys = []

    for key, value in state_dict.items():
        clean_key = _clean_state_key(key)
        if clean_key in model_keys:
            remapped_state[clean_key] = value
            continue

        base_layer_key = _find_base_layer_key(clean_key, model_keys)
        if base_layer_key is not None:
            remapped_state[base_layer_key] = value
            continue

        dropped_keys.append(clean_key)

    incompatible = model.load_state_dict(remapped_state, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys) + dropped_keys

    if strict:
        real_missing = [key for key in missing_keys if "lora_" not in key]
        if real_missing or unexpected_keys:
            raise RuntimeError(
                "Error(s) in loading state_dict with LoRA compatibility:\n"
                f"Missing keys: {real_missing}\n"
                f"Unexpected keys: {unexpected_keys}"
            )
    return missing_keys, unexpected_keys


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ActionEmbedder(nn.Module):
    """
    Embeds action xy into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        hsize = hidden_size//3
        self.x_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.y_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.angle_emb = TimestepEmbedder(hidden_size -2*hsize, frequency_embedding_size)

    def forward(self, xya):
        return torch.cat([self.x_emb(xya[...,0:1]), self.y_emb(xya[...,1:2]), self.angle_emb(xya[...,2:3])], dim=-1)

#################################################################################
#                                 Core CDiT Model                                #
#################################################################################

class CDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c, x_cond):
        shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(11, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
        x = x + gate_ca_x.unsqueeze(1) * self.cttn(query=modulate(self.norm2(x), shift_ca_x, scale_ca_x), key=x_cond_norm, value=x_cond_norm, need_weights=False)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        context_size=2,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ActionEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(self.context_size + 1, num_patches, hidden_size), requires_grad=True) # for context and for predicted frame
        self.blocks = nn.ModuleList([CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)


        # Initialize action embedding:
        nn.init.normal_(self.y_embedder.x_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.x_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.y_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.angle_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.angle_emb.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)
            
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, x_cond, rel_t):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed[self.context_size:]
        x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(0, (x_cond.shape[0], x_cond.shape[1])) + self.pos_embed[:self.context_size]  # (N, T, D), where T = H * W / patch_size ** 2.flatten(1, 2)
        x_cond = x_cond.flatten(1, 2)
        t = self.t_embedder(t[..., None])
        y = self.y_embedder(y) 
        time_emb = self.time_embedder(rel_t[..., None])
        c = t + time_emb + y # if training on unlabeled data, dont add y.

        for block in self.blocks:
            x = block(x, c, x_cond)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   CDiT Configs                                  #
#################################################################################

def CDiT_XL_2(**kwargs):
    return CDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def CDiT_L_2(**kwargs):
    return CDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def CDiT_B_2(**kwargs):
    return CDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def CDiT_S_2(**kwargs):
    return CDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


CDiT_models = {
    'CDiT-XL/2': CDiT_XL_2, 
    'CDiT-L/2':  CDiT_L_2, 
    'CDiT-B/2':  CDiT_B_2, 
    'CDiT-S/2':  CDiT_S_2
}
