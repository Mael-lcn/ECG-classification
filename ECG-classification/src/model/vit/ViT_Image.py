
import torch
import torch.nn as nn
 
from .vit_modules import PatchEmbed, TransformerBlock
 
 
class ViT_Image(nn.Module):
    """
    Mini ViT on rendered ECG image representation. Uses create_image_12leads_perchan() 
    to convert the raw 1D ECG signal into a 12-channel 2D image (one channel per lead), 
    then we can apply a standard ViT encoder identical to ViT_TimeFreq.
    forward(x, batch_mask=None) -> logits (batch, num_classes)
    """
 
    def __init__(
        self,
        num_classes: int = 27, # Number of output classes (27)
        in_channels: int = 12, # Number of ECG leads = number of image channels (12)
        img_size: tuple = (518, 518), # set this to 518 for matching generate_image output...
        segment_size: int = 4000, # set to default to match geenrate_image
        patch_size: tuple = (37, 37), # (14, 14) 
        d_model: int = 128, # Transformer embedding dimension
        num_heads: int = 4, # d_model % num_heads = 0
        num_layers: int = 4, # nb transformer blocks
        mlp_ratio: float = 4.0, # mlp hidden dim = d_model x mlp_ratio
        dropout: float = 0.1,
        emb_dropout: float = 0.1, # dropout on token embeddings before encoder
        **kwargs
    ):
        super().__init__()
 
        self.img_size     = img_size
        self.segment_size = segment_size
        self.in_channels  = in_channels
 
        # Check: patch_size divides img_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, (
            f"patch_size {patch_size} must divide img_size {img_size} evenly. "
            f"For img_size=518: valid patch sizes include 2,7,14,37,74,259,518."
        )
 
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=d_model
        )

        # Same as vit_timefreq
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, 2048, d_model))  # 2048 > max patches
        self.emb_dropout = nn.Dropout(emb_dropout)
 
        # Transfoemer encoder
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
 
        self.init_weights()
 
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        h_patches = img_size[0] // patch_size[0]
        w_patches = img_size[1] // patch_size[1]
        print(f"[ViT_Image] {n_params/1e6:.2f}M parameters | "
              f"d_model={d_model}, heads={num_heads}, layers={num_layers}, "
              f"patch={patch_size}, grid={h_patches}x{w_patches}={h_patches*w_patches} patches")
 
    def init_weights(self):
        """ 
        Same as ViT_TimeFreq
        """
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
 
    def forward(self, x, batch_mask=None, **kwargs):
        """
        Basically the same pipeline as ViT_TimeFreq
        
        Args:
            x: (batch, num_segments, 12, H, W) uint8 from TurboDataset_Img
            or (batch, 12, H, W) if already preprocessed???
        """
        # take first segment
        if x.dim() == 5:
            x = x[:, 0] # (batch, 12, H, W)

        x = x.float() / 255.0

        batch = x.shape[0]
        x = self.patch_embed(x) # (batch, num_patches, d_model)
        n_patches = x.shape[1] # number of patches

        cls = self.cls_token.expand(batch, -1, -1) # (batch, 1, d_model)
        x   = torch.cat([cls, x], dim=1) # (batch, 1+n_patches, d_model), prepends one CLS token per sample in the batch
        x   = x + self.pos_embed[:, :n_patches + 1, :] # +1 for the CLS token at position 0
        x   = self.emb_dropout(x)
        x   = self.blocks(x) # (batch, 1+n_patches, d_model)

        cls_out = self.norm(x[:, 0]) # (batch, d_model) here we only take the CLS token

        return self.head(cls_out) # (batch, num_classes)
 
