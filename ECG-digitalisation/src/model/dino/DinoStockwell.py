import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel



class DinoStockwell(nn.Module):
    """
    Architecture Hiérarchique Optimisée : 
    1. Chunking & Squeeze : On ne traite que les segments avec du signal.
    2. STFT Multicanal : Transformation Temps -> Fréquence.
    3. Lead Projection : Compression 12 leads -> 3 canaux RGB via CNN 1x1.
    4. DINO Backbone : Extraction de features visuelles.
    5. Temporal Transformer : Analyse du rythme global entre les segments.
    """
    def __init__(self, num_classes=27, n_fft=256, chunk_size=4000):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_fft = n_fft

        # Fenêtre de Hann et Projecteur Leads (12 -> 3)
        self.register_buffer('window', torch.hann_window(n_fft))
        self.lead_projector = nn.Conv2d(12, 3, kernel_size=1)

        # Normalisation pour aligner les spectrogrammes sur les stats ImageNet de DINO
        self.input_norm = nn.BatchNorm2d(3)

        # Backbone DINOv3
        repo_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        self.backbone = AutoModel.from_pretrained(
            repo_id, local_files_only=True, attn_implementation="sdpa"
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = self.backbone.config.hidden_size

        self.max_chunks = 20
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_chunks, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.head = nn.Linear(self.embed_dim, num_classes)

    def _get_stft_image(self, x_chunks_valid):
        """
        Transforme les segments de signal en images RGB 224x224.
        """
        N, L, T = x_chunks_valid.shape

        # 1. Aplatir pour STFT
        x_flat = x_chunks_valid.flatten(0, 1)

        # 2. STFT -> Magnitude -> Log
        stft = torch.stft(x_flat, n_fft=self.n_fft, hop_length=64, 
                          window=self.window, return_complex=True, center=True)
        s_mag = torch.log1p(torch.abs(stft))

        # 3. Restaurer la dimension des leads
        s_mag = s_mag.view(N, L, s_mag.shape[-2], s_mag.shape[-1])

        # 4. Compression spatiale (12 leads -> 3 RGB) et Redimensionnement ViT
        s_rgb = self.lead_projector(s_mag)
        imgs = F.interpolate(s_rgb, size=(224, 224), mode='bilinear')

        return self.input_norm(imgs)


    def forward(self, x, batch_mask):
        B, L, T = x.shape

        # 1. Chunking & gestion du padding
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        pad_size = (num_chunks * self.chunk_size) - T
        
        if pad_size > 0:
            x = F.pad(x, (0, pad_size))
            # batch_mask est [B, T], on pad la dimension T avec des 0.0
            batch_mask = F.pad(batch_mask, (0, pad_size), value=0.0)

        x_chunks = x.view(B, num_chunks, 12, self.chunk_size)

        # Extraire 1 valeur par chunk et forcer le type booléen
        mask_chunks = batch_mask[:, ::self.chunk_size].bool()
        flat_mask = mask_chunks.reshape(-1) 

        # Feature extraction
        x_flat = x_chunks.reshape(-1, 12, self.chunk_size)
        valid_x = x_flat[flat_mask] 

        if valid_x.size(0) > 0:
            x_vision = self._get_stft_image(valid_x)
            outputs = self.backbone(x_vision)
            valid_features = outputs.last_hidden_state[:, 0, :] # CLS token

            full_features = torch.zeros(B * num_chunks, self.embed_dim, 
                                      device=x.device, dtype=valid_features.dtype)
            full_features[flat_mask] = valid_features
        else:
            full_features = torch.zeros(B * num_chunks, self.embed_dim, device=x.device)

        # Raisonnement temporel global
        chunk_features = full_features.view(B, num_chunks, -1)
        chunk_features = chunk_features + self.pos_embedding[:, :num_chunks, :]

        temporal_mask = ~mask_chunks 
        refined_features = self.temporal_encoder(chunk_features, src_key_padding_mask=temporal_mask)

        # Pooling final (Moyenne des segments valides uniquement)
        # On repasse mask_chunks en float pour les multiplications
        mask_chunks_float = mask_chunks.float().unsqueeze(-1)

        refined_features = refined_features * mask_chunks_float
        sum_features = refined_features.sum(dim=1)
        count_features = mask_chunks_float.sum(dim=1) + 1e-8

        return self.head(sum_features / count_features)

