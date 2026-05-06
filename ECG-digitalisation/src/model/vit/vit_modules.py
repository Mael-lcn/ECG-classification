"""
ViT building blocks used by both ViT_TimeFreq and ViT_Image (i haven't implemented yet)
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Splits a 2D feature map into non-overlapping patches and projects each patch to a d_model-dimensional 
    embedding vector.

    Args:
        in_channels: Number of input channels (leads for spectrogram, 12 if we use image as input)
        patch_size: (freq_patch, time_patch) or (h_patch, w_patch)
        d_model: Output embedding dimension

    Input: (batch, in_channels, H, W)
    Output: (batch, num_patches, d_model)
    """

    def __init__(self, in_channels: int, patch_size: tuple, d_model: int):
        super().__init__()
        self.patch_size = patch_size

        # Equivalent to manually splitting + linear projection but this way is GPU-efficient.
        self.proj = nn.Conv2d(
            in_channels, 
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.proj(x) # (batch, d_model, H//ph, W//pw)
        x = x.flatten(2) # (batch, d_model, num_patches)
        x = x.transpose(1, 2) # (batch, num_patches, d_model)
        x = self.norm(x)
        return x


class SelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention (Q, K, V)

    Args:
        d_model: Embedding dimension.
        num_heads: Number of attention heads. d_model % num_heads must == 0.
        dropout: Dropout on attention weights.

    Input: (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # size of each head's subspace
        self.scale = self.head_dim ** -0.5 # 1 / sqrt(head_dim), scale down the attention scores to avoid vanishing gradient

        # concatenated along the last dimension and split it later (GPU efficiency)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model) # mixes the isolated heads tgt
        self.attn_drop = nn.Dropout(dropout) # applied on the attention weights 

    def forward(self, x):
        batch, seq_len, d_model = x.shape # seq_len = number of patches

        qkv = self.qkv(x) # (batch, seq_len, 3*d_model)

        # Splits the big vector into Q, K, V and further splits each into num_heads slices, 
        # each head has its own head_dim-sized Q, K, V.
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv.unbind(0) # each: (batch, heads, seq_len, head_dim)

        # dot product between every pair of patches that just gives us raw scores then scale
        attn = (q @ k.transpose(-2, -1)) * self.scale # (batch, heads, seq_len, seq_len)
        attn = attn.softmax(dim=-1) # applied on patch dimension (how much should patch i attend to patch j) 
        attn = self.attn_drop(attn)

        # weighted sum of values for each patch, swap heads and seq_len and concatenate all heads back into one vector 
        x = (attn @ v).transpose(1, 2).reshape(batch, seq_len, d_model) # (batch, seq_len, d_model)
        x = self.proj(x) # mixes the isolated heads tgt
        return x
    

class TransformerBlock(nn.Module):
    """
    Standard ViT encoder block

    Args:
        d_model: Transformer embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim = d_model * mlp_ratio (4 is standard ViT)
        dropout: Applied in both attention and MLP
    """
    def __init__(self, d_model: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = SelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(), # Gaussian CDF smooth gradients for large number of layers....
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm is more stable than post-norm for small models trained from scratch
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x