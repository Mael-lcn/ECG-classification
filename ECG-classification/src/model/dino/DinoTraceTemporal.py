import torch
import torch.nn as nn
from transformers import AutoModel



class DinoTraceTemporal(nn.Module):
    """
    Modèle DINOv3 (via Hugging Face Cache) avec gestion VRAM stricte (Micro-Batching),
    suivi d'une Attention Temporelle, Max Pooling et classification MLP.
    """
    def __init__(
        self, 
        num_classes=26, 
        D_hidden_dim=512,
        D_dropout_classifier=0.4,
        sub_batch_size=16,
        D_nhead=8
    ):
        super().__init__()
        self.sub_batch_size = sub_batch_size

        # 1. Chargement du Backbone DINO depuis le cache standard Hugging Face
        # L'identifiant officiel du repo
        repo_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"

        self.backbone = AutoModel.from_pretrained(repo_id, local_files_only=True, attn_implementation="sdpa")        
        embed_dim = self.backbone.config.hidden_size

        # Gel total du backbone pour ne pas exploser la RAM
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Buffers de Normalisation
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Attention Temporelle (Self/Cross-Attention sur la séquence)
        self.temporal_attention = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=D_nhead,
            dim_feedforward=embed_dim * 2,
            dropout=0.15,
            batch_first=True
        )

        # Tête de Classification finale (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, D_hidden_dim),
            nn.BatchNorm1d(D_hidden_dim),
            nn.ReLU(),
            nn.Dropout(D_dropout_classifier),
            nn.Linear(D_hidden_dim, num_classes)
        )

    def forward(self, x, batch_mask):
        """
        Args:
            x (torch.Tensor): [Batch, Segments, 3, H, W] - Images uint8 ou float
        """
        B, S, C, H, W = x.shape
        # Aplatissement 5D -> 4D pour traitement indépendant par DINO
        x_flat = x.view(B * S, C, H, W)

        all_features = []

        with torch.no_grad():
            for i in range(0, B * S, self.sub_batch_size):
                chunk = x_flat[i : i + self.sub_batch_size]

                # Conversion et Normalisation au dernier moment
                if chunk.dtype == torch.uint8:
                    chunk = chunk.float() / 255.0
                chunk = (chunk - self.mean) / self.std

                # Extraction DINO via Hugging Face
                outputs = self.backbone(pixel_values=chunk)
                
                # Le modèle HF renvoie la séquence de patchs. 
                # On prend le token [CLS] (index 0) qui représente l'image entière condensée.
                feat = outputs.last_hidden_state[:, 0, :]

                all_features.append(feat)

        # Recombinaison temporelle des blocs
        features = torch.cat(all_features, dim=0).view(B, S, -1)

        # Attention Temporelle
        attn_out = self.temporal_attention(features)

        # Agrégation MAX
        final_rep, _ = torch.max(attn_out, dim=1) 

        # MLP Classifieur
        logits = self.classifier(final_rep)

        return logits
