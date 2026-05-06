import torch
import torch.nn as nn



class EncoderLayerBN(nn.Module):
    """
        Initialise une couche d'encodage comprenant optionnellement un sous-échantillonnage, 
        suivi d'une double convolution avec normalisation par lot et activation LeakyReLU.

        Args:
            ch_in (int): Nombre de canaux spatiaux en entrée.
            ch_out (int): Nombre de canaux spatiaux en sortie.
            kernel_size (int): Taille du noyau de convolution.
            padding (int): Espacement appliqué aux bords lors de la convolution.
            pooling (bool): Détermine si une opération de Max Pooling est appliquée en entrée.
            dropout (float): Probabilité de mise à zéro des éléments pour la régularisation.
    """
    def __init__(self, ch_in, ch_out, kernel_size, padding, pooling, dropout):
        super(EncoderLayerBN, self).__init__()

        if pooling:
            self.pooling = nn.MaxPool2d(2)
        else:
            self.pooling = None

        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        return self.block(x)


class DecoderLayerBN(nn.Module):
    """
        Initialise une couche de décodage avec suréchantillonnage spatial, intégration 
        des connexions résiduelles (skip connections) et double convolution.

        Args:
            ch_in (int): Nombre de canaux spatiaux en entrée.
            ch_out (int): Nombre de canaux spatiaux en sortie.
            kernel_size (int): Taille du noyau de convolution.
            padding (int): Espacement appliqué aux bords lors de la convolution.
            dropout (float): Probabilité de mise à zéro des éléments pour la régularisation.
            skip_mode (str): Stratégie d'intégration des caractéristiques de l'encodeur ("concat", "add", "none").
            upsampling_mode (str): Méthode d'agrandissement spatial ("transpose" ou "bilinear").
            cropping (bool): Détermine si un rognage central doit être appliqué aux skip features.
    """
    def __init__(self, ch_in, ch_out, kernel_size, padding, dropout, skip_mode="concat", upsampling_mode="transpose", cropping=False):
        super(DecoderLayerBN, self).__init__()

        self.cropping = cropping
        self.skip_mode = skip_mode
        self.upsampling_mode = upsampling_mode

        if self.upsampling_mode == "transpose":
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

        if self.skip_mode == "concat":
            ch_hidden = ch_out + ch_out
        elif self.skip_mode == "add":
            ch_hidden = ch_out
        elif self.skip_mode == "none":
            ch_hidden = ch_out

        self.block = nn.Sequential(
            nn.Conv2d(ch_hidden, ch_out, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def crop(self, x, cropping_size):
        """
        Rogne symétriquement un tenseur spatialement pour correspondre à une dimension cible.

        Args:
            x (torch.Tensor): Tenseur de caractéristiques à rogner.
            cropping_size (torch.Tensor): Tenseur contenant le nombre de pixels à retirer sur la hauteur et la largeur.

        Returns:
            torch.Tensor: Tenseur rogné aux dimensions ajustées.
        """
        h_crop, w_crop = cropping_size[0].item(), cropping_size[1].item()
        if h_crop == 0 and w_crop == 0:
            return x

        h_end = -h_crop if h_crop > 0 else None
        w_end = -w_crop if w_crop > 0 else None

        return x[:, :, h_crop:h_end, w_crop:w_end]

    def forward(self,x, skip_features):
        if self.upsampling_mode == "transpose":
            x = self.up(x)
        else:
            x = self.up(x)
            x = self.conv(x)

        if self.cropping:
            cropping_size = (torch.tensor(skip_features.shape[2:]) - torch.tensor(x.shape[2:]))//2
            skip_features = self.crop(skip_features, cropping_size)

        if self.skip_mode == "concat":
            x = self.block(torch.cat((x, skip_features), 1))
        elif self.skip_mode == "add":
            x = self.block(x + skip_features)
        elif self.skip_mode == "none":
            x = self.block(x)

        return x



class UNet2d(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 encoder_layer=EncoderLayerBN,
                 decoder_layer=DecoderLayerBN,
                 hidden_dims=[64,128,256,512,1024],
                 kernel_size=3,
                 padding_mode="valid",
                 skip_mode="none",
                 upsampling_mode="transpose",
                 dropout=0,
                 ):
        """
        Initialise l'architecture U-Net ou Autoencodeur convolutif 2D.

        Args:
            input_dim (int): Nombre de canaux de l'image d'entrée.
            output_dim (int): Nombre de canaux de l'image reconstruite en sortie.
            encoder_layer (class): Classe ou fonction générant un bloc d'encodage.
            decoder_layer (class): Classe ou fonction générant un bloc de décodage.
            hidden_dims (list of int): Liste définissant la progression du nombre de canaux par niveau de profondeur.
            kernel_size (int): Taille globale des noyaux convolutifs du réseau.
            padding_mode (str): Mode de gestion des bords ("same" ou "valid").
            skip_mode (str): Méthode de fusion pour les connexions résiduelles ("concat", "add", "none").
            upsampling_mode (str): Méthode d'agrandissement pour le décodeur ("transpose" ou "bilinear").
            dropout (float): Taux de régularisation appliqué aux couches les plus profondes.
        """
        super(UNet2d, self).__init__()

        assert len(hidden_dims) > 0, "UNet2d requires at least one hidden layer"
        assert padding_mode in ["same", "valid"], f"Padding mode has to be either 'same' or 'valid' but got '{padding_mode}'"

        self.padding_mode = padding_mode

        cropping = True if padding_mode == "valid" else False
        padding = 0 if padding_mode == "valid" else kernel_size//2

        # Assembling the encoder
        encoder = []
        for i in range(len(hidden_dims)):
            if i == 0:
                ch_in = input_dim
                ch_out = hidden_dims[i]
                encoder.append(encoder_layer(ch_in, ch_out, kernel_size=kernel_size, padding=padding, pooling=False, dropout=0))
            elif i == (len(hidden_dims) - 1):
                ch_in = hidden_dims[i-1]
                ch_out = hidden_dims[i]
                encoder.append(encoder_layer(ch_in, ch_out, kernel_size=kernel_size, padding=padding, pooling=True, dropout=dropout))
            else:
                ch_in = hidden_dims[i-1]
                ch_out = hidden_dims[i]
                encoder.append(encoder_layer(ch_in, ch_out, kernel_size=kernel_size, padding=padding, pooling=True, dropout=0))
        self.encoder = nn.ModuleList(encoder)

        #Assembling the decoder
        decoder = []

        # Reversing the order of the hidden dims, since the decoder reduces the number of channels
        hidden_dims_rev = hidden_dims[::-1]

        for i in range(len(hidden_dims_rev) - 1):
            ch_in = hidden_dims_rev[i]
            ch_out = hidden_dims_rev[i+1]
            decoder.append(decoder_layer(ch_in, ch_out, kernel_size=kernel_size, padding=padding, dropout=0, skip_mode=skip_mode, upsampling_mode=upsampling_mode, cropping=cropping))
        self.decoder = nn.ModuleList(decoder)

        # Creating final 1x1 convolution
        self.final_conv = nn.Conv2d(hidden_dims[0], output_dim, kernel_size=1, stride=1, padding=0)
        self.final_act = nn.Tanh()


    def encode(self, x):
        """Compresse l'image dans l'espace latent (bottleneck)."""
        skip_features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_features.insert(0, x)

        # Le premier élément est le bottleneck, le reste sont les skips
        bottleneck = skip_features[0]
        skips = skip_features[1:]
        return bottleneck, skips

    def decode(self, z, skip_features=None):
        """Reconstruit l'image à partir d'un vecteur latent z."""
        x = z
        for i, decoder_layer in enumerate(self.decoder):
            # Si skip_mode est "none", skip peut être None ou des zéros
            skip = skip_features[i] if skip_features is not None else None
            x = decoder_layer(x, skip)

        x = self.final_conv(x)
        return self.final_act(x)

    def forward(self, x):
        z, skips = self.encode(x)
        return self.decode(z, skips)


def weights_init(m):
    """
    Initialise les poids d'un module PyTorch selon des heuristiques standardisées.
    Utilise l'initialisation Kaiming Normal pour les convolutions et des constantes pour les normalisations.

    Args:
        m (nn.Module): Module PyTorch dont les poids doivent être initialisés.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.2)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
