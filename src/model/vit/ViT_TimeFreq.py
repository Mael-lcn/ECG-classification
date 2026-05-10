import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F

from .vit_modules import PatchEmbed, TransformerBlock


class ViT_TimeFreq(nn.Module):
    """
    Mini ViT on time-frequency representation. Treats each ECG lead as a separate input channel 
    (like RGB channels in image ViTs), so the spectrogram has shape (batch, 12_leads, freq, time)

    Interface is identical to CNN_TimeFreq:
        forward(x, batch_mask=None) -> logits (batch, num_classes)

    Args:
        num_classes: Number of output classes (27)
        in_channels: Number of ECG leads (12)
        n_fft: FFT size -> freq bins = n_fft // 2 + 1
        hop_length: Spectrogram hop (controls time resolution)
        win_length: Spectrogram window size
        patch_size: (freq_patch, time_patch). Must divide (freq, time) evenly
                       Default (5, 5) works for n_fft=128, hop=64
        d_model: Transformer embedding dimension
        num_heads: Attention heads. d_model % num_heads must == 0
        num_layers: Number of TransformerBlocks stacked
        mlp_ratio: MLP hidden dim = d_model x mlp_ratio
        dropout: Dropout in attention and MLP
        emb_dropout: Dropout on token embeddings before encoder
    """

    def __init__(
        self,
        num_classes: int = 27,
        in_channels: int = 12,
        n_fft: int = 128,
        hop_length: int = 64,
        win_length: int = 128,
        patch_size: tuple = (5, 5),
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        # Converts raw 1D signal to 2D power spectrogram
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=2.0  # returns power spectrogram
        )

        # Tokenization: takes the 2D spectrogram (batch, 12, freq, time) and cuts it into patches, 
        # projecting each patch into a d_model dimensional vector.
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=d_model
        )

        # Learnable vector used to aggregate global information from all patches through attention
        # So instead of using average pooling we use this to aggregrate the information across all patch for a single example
        # At each block the CLS token:
        #    1. Attends to all patches — collects relevant information
        #    2. Gets updated via MLP — processes what it collected
        #    3. Passes to the next block — progressively richer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # add a learnable position signal to each token so the model knows where each patch is in the spectrogram
        self.pos_embed = nn.Parameter(torch.zeros(1, 2048, d_model))
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Transformer encoder (one block = attention + MLP)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ViT_TimeFreq] {n_params/1e6:.2f}M parameters | "
              f"d_model={d_model}, heads={num_heads}, layers={num_layers}, "
              f"patch={patch_size}")

    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear): # randomly initialized
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm): # starts as identity
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d): # i've never hear abt this but it is fr gelu??
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def log_spectrogram(self, x):
        B, L, T = x.shape
        
        # Flatten leads into batch
        x_flat = x.reshape(B * L, T)
        
        # Optimization: Only move to CPU if we have to. 
        # But for now, we keep your CPU-offload fix for the driver bug.
        orig_device = x.device
        x_cpu = x_flat.to('cpu', dtype=torch.float32)
        self.spectrogram.to('cpu')
        
        with torch.no_grad():
            spec = self.spectrogram(x_cpu) 
            spec = torch.clamp(spec, min=1e-10)
            spec = torch.log(spec)
            # Normalize
            mean = spec.mean(dim=(-2, -1), keepdim=True)
            std = spec.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
            spec = (spec - mean) / std
        
        # Return to GPU
        x = spec.to(orig_device)
        
        freq_dim, time_dim = x.shape[-2], x.shape[-1]
        
        # GEOMETRY CHECK: 
        # If time_dim < patch_size, we must pad or the Conv2d will crash
        ph, pw = self.patch_embed.patch_size
        if time_dim < pw or freq_dim < ph:
            pad_h = max(0, ph - freq_dim)
            pad_w = max(0, pw - time_dim)
            x = F.pad(x, (0, pad_w, 0, pad_h))
            
        return x.reshape(B, L, x.shape[-2], x.shape[-1])

    def forward(self, x, **kwargs):
        """
        Args:
            x: (batch, n_leads, n_samples) -- raw ECG signal
            batch_mask: ignored, kept for train.py compatibility

        Returns:
            logits: (batch, num_classes)
        """
        B = x.shape[0]

        x = self.log_spectrogram(x) # (B, leads, freq, time)
        x = self.patch_embed(x) # (B, num_patches, d_model)
        N = x.shape[1] # number of patches

        cls = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1) # (B, 1+N, d_model) prepends one CLS token per sample in the batch
        x = x + self.pos_embed[:, :N+1, :] # +1 for the CLS token at position 0
        x = self.emb_dropout(x)
        x = self.blocks(x) # (B, 1+N, d_model)

        cls_out = self.norm(x[:, 0]) # (B, d_model) here we only take the CLS token

        return self.head(cls_out) # (B, num_classes)
