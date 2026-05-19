import argparse
import inspect
import multiprocessing

import os
import sys

project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(project_root))

# All models
from model.cnn.Cnn import CNN
from model.cnn.Cnn_TimeFreq import CNN_TimeFreq
from model.cnn.Cnn_Image import CNN_Image

from PatchTST_CrossAtt import PatchTST_CrossAtt

from model.utils.generate_image import create_image_12leads_perchan, create_image_12leads_together
from vit import ViT_TimeFreq, ViT_Image
from dino import DinoTraceTemporal, DinoStockwell
from mamba import Mamba2_time_series

# All dataset
from torch.utils.data import DataLoader
from TurboDataset import TurboDataset
from TurboDataset_Img import TurboDataset_Img



# Defini un registre de tout les modèles
model_classes = [
    CNN, 
    CNN_TimeFreq,
    CNN_Image,

    PatchTST_CrossAtt, 

    DinoTraceTemporal, 
    DinoStockwell, 

    ViT_TimeFreq,
    ViT_Image,

    Mamba2_time_series
]

DATASET_MAPPING = {
    'CNN_Image': (TurboDataset_Img, create_image_12leads_perchan),
    'ViT_Image': (TurboDataset_Img, create_image_12leads_perchan),
    'DinoTraceTemporal': (TurboDataset_Img, create_image_12leads_together),
    'DEFAULT': (TurboDataset, None)
}


MODEL_REGISTRY = {cls.__name__: cls for cls in model_classes}


def get_shared_parser():
    """Retourne un parser avec les arguments communs à Train et Eval."""
    parser = argparse.ArgumentParser(add_help=False)

    # --- 1. PARAMÈTRES SYSTÈME ET DONNÉES ---
    group_sys = parser.add_argument_group("Configuration Système & Fichiers")
    group_sys.add_argument('--output', type=str, default='../../../output/',
                           help="Dossier de sortie standard")
    parser.add_argument('--checkpoint_dir', default="../../../checkpoints/", help="Dossier où sauvegarder les poids (.pt)")

    group_sys.add_argument('--class_map', type=str, default='../../ressources/final_class.json',
                           help="Chemin JSON mappant les indices aux noms de classes")    
    group_sys.add_argument('-w', '--workers', type=int, default=min(max(4, multiprocessing.cpu_count()-1), 8),
                           help="Nombre de processus pour charger les données")
    group_sys.add_argument('--gpu', type=int, default=0,
                        help="index du GPU a utiliser (config HPC)")

    # --- 2. Paramètres du dataloading communs ---
    group_train = parser.add_argument_group("Hyperparamètres Communs du Dataloader/Inférence")
    group_train.add_argument('--batch_size_theoric', type=int, default=64, help="Taille du batch de MAJ du gradient")
    group_train.add_argument('--batch_size_accumulat', type=int, default=64, help="Taille du batch d'inference")

    group_train.add_argument('--mega_batch_factor', type=int, default=16,
                             help="Granularité du tri. Haut = padding optimisé, Bas = + d'aléatoire")
    group_train.add_argument('--use_static_padding', action='store_true', default=False,
                             help="Force une taille de padding fixe (universelle)")
    group_train.add_argument('--not_use_amp', action='store_true', default=False,
                             help="Désactive l'Automatic Mixed Precision (AMP). Passe en FP32.")
    group_train.add_argument('--image_width', type=int, default=512,
                             help="Largeur de l'image d'entrée générée (en pixels).")
    group_train.add_argument('--image_height', type=int, default=512,
                             help="Hauteur de l'image d'entrée générée (en pixels).")

     # --- 3. Choix du modèle ---
    group_model = parser.add_argument_group("Sélection du Modèle")
    group_model.add_argument('-m', '--model_name', type=str, required=True, choices=MODEL_REGISTRY.keys(),
                             help="Nom du modèle à utiliser")

    # --- 4. Architecture partagés par CNN et CNN_TimeFreq ---
    group_arch = parser.add_argument_group("Hyperparamètres de l'Architecture (Communs)")
    group_arch.add_argument('--num_classes', type=int, default=27, help="Nombre de classes de sortie")
    group_arch.add_argument('--in_channels', type=int, default=12, help="Nombre de dérivations ECG (leads)")
    group_arch.add_argument('--ch1', type=int, default=32, help="Filtres de la 1ère couche convolutive")
    group_arch.add_argument('--ch2', type=int, default=64, help="Filtres de la 2ème couche convolutive")
    group_arch.add_argument('--ch3', type=int, default=128, help="Filtres de la 3ème couche convolutive")
    group_arch.add_argument('--dropout', type=float, default=0.5, help="Taux de dropout")
    group_arch.add_argument('--no_batchnorm', dest='use_batchnorm', action='store_false', default=True,
                            help="Désactive le Batch Normalization (activé par défaut)")
    group_arch.add_argument('--use_fcnn', action='store_true', default=False,
                            help="Active l'architecture FCNN (fenêtre glissante) au lieu de CNN standard")

    # --- 5. Architecture spécifique : CNN 1D ---
    group_cnn1d = parser.add_argument_group("Spécifique au CNN Standard (1D)")
    group_cnn1d.add_argument('--kernel_size', type=int, default=3, help="Taille du kernel de convolution (1D)")
    group_cnn1d.add_argument('--window_size1D', type=int, default=4000, 
                             help="Taille de la fenêtre pour le classifieur FCNN 1D (kernel_size du classifieur)")

    # --- 6. Architecture spécifique : CNN TimeFreq ---
    group_timefreq = parser.add_argument_group("Spécifique au CNN Time-Frequency (Spectrogramme)")
    group_timefreq.add_argument('--n_fft', type=int, default=128, help="Points utilisés pour la transformée de Fourier")
    group_timefreq.add_argument('--hop_length', type=int, default=64, help="Pas de glissement de la fenêtre (stride)")
    group_timefreq.add_argument('--win_length', type=int, default=128, help="Taille de la fenêtre en échantillons")
    group_timefreq.add_argument('--window_size2D', type=int, nargs='+', default=[4, 4],
                                help="Taille de la fenêtre (freq, time) pour le classifieur FCNN 2D. Ex: --window_size_2d 4 4")

    # --- 7. Architecture spécifique : Transformer (PatchTST) ---
    group_patchtst = parser.add_argument_group("Spécifique au Transformer (PatchTST)")
    # On aligne le default sur les 2000 points du modèle original
    group_patchtst.add_argument('--PT_context_length', type=int, default=2000, help="Taille de la fenêtre temporelle en entrée")
    group_patchtst.add_argument('--PT_patch_length', type=int, default=40, help="Taille d'un patch (ex: 40 points = 100ms)")
    group_patchtst.add_argument('--PT_patch_stride', type=int, default=20, help="Chevauchement entre les patchs")
    group_patchtst.add_argument('--PT_d_model', type=int, default=128, help="Dimension interne du Transformer")
    group_patchtst.add_argument('--PT_num_heads', type=int, default=8, help="Nombre de têtes d'attention")
    group_patchtst.add_argument('--PT_cross_att_heads', type=int, default=8, help="Nombre de têtes pour la cross-attention")
    group_patchtst.add_argument('--PT_encoder_layers', type=int, default=3, help="Profondeur du Transformer")
    group_patchtst.add_argument('--PT_revin', action='store_true', default=False, help="Active la Reversible Instance Normalization")
    group_patchtst.add_argument('--no_cross_att', dest='PT_use_cross_att', action='store_false', default=True, 
                                help="Désactive la Multi-Head Attention entre les canaux")

    # --- 8. Architecture spécifique : Transformer (ViT_TimeFreq)  ---
    group_vit = parser.add_argument_group("Spécifique au ViT")
    group_vit.add_argument('--num_layers', type=int, default=4,
                            help="Nombre de blocs Transformer dans le ViT")
    group_vit.add_argument('--mlp_ratio', type=float, default=4.0,
                            help="Ratio MLP hidden dim / d_model")
    group_vit.add_argument('--emb_dropout', type=float, default=0.1,
                            help="Dropout sur les embeddings avant l'encodeur")
    group_vit.add_argument('--patch_size', type=int, nargs='+', default=[5, 5], help="Taille des patches 2D. Ex: --patch_size 5 5")

    # --- 9. Architecture spécifique : CNN Image  ---
    group_cnnImage = parser.add_argument_group("Spécifique au CNN image")
    group_cnnImage.add_argument('--cnn_mode', type=str, default='square', choices=['square', 'rectangle'],
                        help="Mode for CNN_Image: 'square' (3x3) or 'rectangle' (half-height)")

    return parser


def build_model(args_namespace):
    """Instancie le modèle en filtrant dynamiquement les arguments."""
    args_dict = vars(args_namespace)
    model_name = args_dict.get('model_name')

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modèle inconnu. Choix : {list(MODEL_REGISTRY.keys())}")

    ModelClass = MODEL_REGISTRY[model_name]

    # Introspection pour ne donner au modèle que ce qu'il demande
    sig = inspect.signature(ModelClass.__init__)
    valid_kwargs = {k: v for k, v in args_dict.items() if k in sig.parameters}

    print(f"Le modèle: {model_name} à bien été instancié")
    
    dataset_info = DATASET_MAPPING.get(model_name, DATASET_MAPPING['DEFAULT'])
    Dataset_fun, gen_fun = dataset_info

    return ModelClass(**valid_kwargs), valid_kwargs, Dataset_fun, gen_fun



def create_dataloader(args, data_path, Dataset_fun, gen_fun=None, is_train=True):
    """
    Crée un DataLoader standardisé pour assurer l'homogénéité entre train, val, et test.
    Gère automatiquement les kwargs du dataset et les optimisations multiprocessing.
    """
    dataset_kwargs = {
        "batch_size": args.batch_size_accumulat,
        "mega_batch_size": args.batch_size_theoric * args.mega_batch_factor,
        "use_static_padding": args.use_static_padding
    }

    if gen_fun is not None:
        dataset_kwargs["generate_img"] = gen_fun
        dataset_kwargs["h"] = args.image_height
        dataset_kwargs["w"] = args.image_width

    # Instanciation du dataset
    dataset = Dataset_fun(data_path=data_path, is_train=is_train, **dataset_kwargs)

    # Création du DataLoader optimisé
    loader = DataLoader(
        dataset,
        batch_size=None,    # None car TurboDataset gère déjà le batching
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None     # Évite un crash si workers=0
    )
    
    return loader
