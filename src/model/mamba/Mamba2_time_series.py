import torch
import torch.nn as nn
from mamba_ssm import Mamba2



class Mamba2_time_series(nn.Module):
    """
    architecture mamba 2 pour la classification de séries temporelles (ex: ecg).
    combine un extracteur local (cnn) et un modèle d'état (mamba 2) profond 
    pour capturer les dépendances à long terme.
    """
    def __init__(
        self, 
        in_channels=12,
        num_classes=27,
        mamba_dim=768,
        num_mamba_layers=4,
        M2_hidden_dim=1024,
        M2_dropout_rate=0.4
    ):
        super().__init__()
        
        # projection initiale via cnn pour réduire la dimension temporelle par 4
        # noyaux larges au début (15) pour capter la morphologie des ondes
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, mamba_dim // 4, kernel_size=15, stride=2, padding=7),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim // 4),
            
            nn.Conv1d(mamba_dim // 4, mamba_dim // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim // 2),
            
            nn.Conv1d(mamba_dim // 2, mamba_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim)
        )
        
        # empilement de plusieurs couches mamba avec normalisation et résidu
        self.mamba_layers = nn.ModuleList([
            nn.ModuleDict({
                'mixer': Mamba2(
                    d_model=mamba_dim,
                    d_state=128,
                    d_conv=4,
                    expand=2,
                    headdim=128
                ),
                'norm': nn.LayerNorm(mamba_dim)
            }) for _ in range(num_mamba_layers)
        ])
        
        # tête de classification
        self.classifier = nn.Sequential(
            nn.Linear(mamba_dim, M2_hidden_dim),
            nn.BatchNorm1d(M2_hidden_dim),
            nn.ReLU(),
            nn.Dropout(M2_dropout_rate),
            nn.Linear(M2_hidden_dim, num_classes)
        )

    def forward(self, x, batch_mask=None, **kwargs):
        """
        passe avant du modèle.
        x: tenseur de forme [batch, canaux, longueur]
        batch_mask: tenseur binaire [batch, longueur] issu de create_attention_mask
        """
        # 1. extraction et réduction temporelle
        x_features = self.input_proj(x)

        # transposition pour le format mamba [batch, longueur, dimension]
        hidden_states = x_features.transpose(1, 2)

        # 2. passage dans les blocs mamba avec connexions résiduelles
        for layer in self.mamba_layers:
            residual = hidden_states
            # application de la normalisation puis du mixeur
            mixed = layer['mixer'](layer['norm'](hidden_states))
            hidden_states = mixed + residual

        # 3. agrégation temporelle avec gestion exacte du masque
        if batch_mask is not None:
            # On force le masque à avoir la taille exacte de la sortie du CNN par interpolation
            mask_downsampled = torch.nn.functional.interpolate(
                batch_mask.unsqueeze(1).float(), 
                size=hidden_states.shape[1], # Taille cible (ex: 734)
                mode='nearest'
            ).squeeze(1)

            # ajout de la dimension pour la diffusion
            mask_downsampled = mask_downsampled.unsqueeze(-1)

            # moyenne pondérée ignorant le padding
            sum_hidden = (hidden_states * mask_downsampled).sum(dim=1)
            valid_lengths = mask_downsampled.sum(dim=1).clamp(min=1e-9)
            final_rep = sum_hidden / valid_lengths
        else:
            final_rep = hidden_states.mean(dim=1)

        # 4. classification finale
        logits = self.classifier(final_rep)
        
        return logits
