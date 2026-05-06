import os, sys
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import bernoulli

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ecg-image-kit', 'ecg-image-generator')))

# Importation des modules externes 
from ecg_plot import ecg_plot
from TurboDataset import TurboDataset


# Fixation des graines pour assurer la reproductibilité des expériences
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def ecg_image_kit_vectoriel(batch_x, batch_lens, h=512, w=512, freq=400, duree_cible=10, is_train=True):
    """
    Interface d'intégration pour la génération d'images avec augmentation.
    
    Traite les signaux bruts, applique un recadrage temporel, génère l'image en mémoire,
    et retourne simultanément l'image générée et le segment de signal correspondant 
    utilisé comme vérité terrain pour la reconstruction.

    Args:
        batch_x (torch.Tensor): Tenseur des signaux bruts [batch_size, 12, length].
        batch_lens (torch.Tensor): Longueurs valides de chaque signal.
        h (int): Hauteur de l'image de sortie.
        w (int): Largeur de l'image de sortie.
        freq (int): Fréquence d'échantillonnage.
        duree_cible (int): Durée de la fenêtre en secondes.
        is_train (bool): Active le caractère aléatoire des augmentations si True.

    Returns:
        tuple: (batch_images_tensor, batch_targets_tensor, num_windows)
    """
    batch_size = batch_x.shape[0]
    points_cibles = freq * duree_cible

    batch_images = []
    batch_targets = []

    # Définition des probabilités d'augmentation (applicables uniquement en entraînement)
    prob_bw = bernoulli(0.2)       
    prob_grid = bernoulli(0.9)     
    prob_dc = bernoulli(1.0)       

    noms_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    configs_mock = {
        'leadNames_12': noms_leads,
        'format_4_by_3': [['I', 'II', 'III'], ['aVR', 'aVL', 'aVF'], ['V1', 'V2', 'V3'], ['V4', 'V5', 'V6']],
        'tickLength': 0.5,
        'tickSize_step': 0.5
    }

    for i in range(batch_size):
        signal = batch_x[i]
        longueur_reelle = batch_lens[i].item()

        # Recadrage temporel : la portion extraite devient la vérité terrain
        if longueur_reelle > points_cibles:
            if is_train:
                start = np.random.randint(0, longueur_reelle - points_cibles)
            else:
                start = 0
            signal_crop = signal[:, start : start + points_cibles]
        elif longueur_reelle < points_cibles:
            padding_size = points_cibles - longueur_reelle
            signal_crop = F.pad(signal[:, :longueur_reelle], (0, padding_size))
        else:
            signal_crop = signal[:, :points_cibles]

        signal_np = signal_crop.cpu().numpy()
        ecg_dict = {noms_leads[j]: signal_np[j] for j in range(12)}
        ecg_dict['fullII'] = signal_np[1] 

        # Détermination des paramètres visuels selon la phase
        if not is_train:
            is_bw = False
            has_grid = True
            has_dc = True
            random_resolution = 200
            random_padding = 0
        else:
            is_bw = prob_bw.rvs()
            has_grid = prob_grid.rvs()
            has_dc = prob_dc.rvs()
            random_resolution = random.choice(range(100, 300))
            random_padding = random.choice(range(0, 2))
            
        grid_style = 'bw' if is_bw else None

        # Rendu de l'image
        image_tensor, _, _ = ecg_plot(
            ecg=ecg_dict, 
            configs=configs_mock,
            sample_rate=freq, 
            columns=4,
            rec_file_name="", output_dir="", pad_inches=random_padding, store_text_bbox=False, full_header_file="",
            resolution=random_resolution, lead_index=noms_leads, full_mode='II', show_lead_name=True,
            show_grid=bool(has_grid), show_dc_pulse=bool(has_dc), style=grid_style, standard_colours=0, print_txt=False,
            lead_length_in_seconds=duree_cible
        )
        
        # Redimensionnement spatial
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = F.interpolate(image_tensor, size=(h, w), mode='bilinear', align_corners=False)
        image_tensor = image_tensor.squeeze(0)

        batch_images.append(image_tensor)
        batch_targets.append(signal_crop)

    batch_images_tensor = torch.stack(batch_images)
    batch_targets_tensor = torch.stack(batch_targets)
    num_windows = torch.ones(batch_size, dtype=torch.long)

    return batch_images_tensor, batch_targets_tensor, num_windows


class TurboDataset_Img(TurboDataset):
    """
    Dataset dynamique de génération d'images.
    Renvoie le batch d'images ainsi que les signaux temporels (cibles) correspondants.
    """
    def __init__(self, data_path, generate_img_func, h=512, w=512, is_train=True, **kwargs):
        super().__init__(data_path, is_train=is_train, **kwargs)
        self.h = h
        self.w = w
        self.generate_img = generate_img_func
        self.is_train = is_train

    def __iter__(self):
        # La classe parente renvoie les signaux initiaux
        for batch_x, batch_y_ignored, batch_lens in super().__iter__():
            
            # La fonction génère l'image ET extrait le signal précis utilisé
            batch_images, batch_targets, num_windows = self.generate_img(
                batch_x, 
                batch_lens, 
                h=self.h, 
                w=self.w,
                is_train=self.is_train
            )
            yield batch_images, batch_targets, num_windows


class StaticImageDataset(Dataset):
    """
    Dataset lisant des données statiques où la vérité terrain est une série temporelle CSV.
    Groupe les images par identifiant d'électrocardiogramme pour éviter la fuite de données.
    """
    def __init__(self, folder_path, h=512, w=512, is_train=True, points_cibles=4000):
        self.folder = folder_path
        self.h = h
        self.w = w
        self.is_train = is_train
        self.points_cibles = points_cibles

        # Regroupement des enregistrements par ID
        self.records = []
        csv_files = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)

        for csv_path in sorted(csv_files):
            base_name = os.path.basename(csv_path).replace('.csv', '')
            dir_name = os.path.dirname(csv_path)

            # Recherche de toutes les variantes visuelles associées à ce signal
            images_associees = glob.glob(os.path.join(dir_name, f"{base_name}-*.*"))
            images_associees = [img for img in images_associees if img.endswith(('.png', '.jpg'))]

            if images_associees:
                self.records.append({
                    'id': base_name,
                    'csv_path': csv_path,
                    'images': sorted(images_associees)
                })

        # Brassage de la liste des dossiers en mode entraînement
        if self.is_train:
            random.Random(42).shuffle(self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # 1. Sélection de l'image (Hasard en entraînement, déterministe '0001' en validation)
        if self.is_train:
            filepath = random.choice(record['images'])
        else:
            # Recherche de l'image de référence -0001, sinon sélection de la première disponible
            filepath = next((img for img in record['images'] if '0001' in os.path.basename(img)), record['images'][0])

        # Chargement de l'image
        img = Image.open(filepath).convert('RGB')
        img = img.resize((self.w, self.h))
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0

        # 2. Chargement de la vérité terrain (Signal CSV)
        # On extrait uniquement les valeurs numériques et on transpose pour avoir [12, Longueur]
        try:
            df = pd.read_csv(record['csv_path'])
            data_num = df.select_dtypes(include=[np.number]).values
            signal_tensor = torch.from_numpy(data_num).float().T
        except Exception:
            # Sécurité en cas de fichier manquant ou corrompu
            signal_tensor = torch.zeros((12, self.points_cibles))

        # Ajustement de la taille du signal pour correspondre aux attentes du modèle
        longueur_reelle = signal_tensor.shape[1]
        if longueur_reelle > self.points_cibles:
            signal_tensor = signal_tensor[:, :self.points_cibles]
        elif longueur_reelle < self.points_cibles:
            padding_size = self.points_cibles - longueur_reelle
            signal_tensor = F.pad(signal_tensor, (0, padding_size))

        num_windows = torch.tensor(1, dtype=torch.long)
        return img_tensor, signal_tensor, num_windows


class MixedDataLoader:
    """
    Gestionnaire global encapsulant l'instanciation des flux statiques et dynamiques.
    Fusionne les lots et maintient l'équilibre pour l'entraînement des gradients.
    """
    def __init__(self, 
                 static_folder, 
                 dynamic_data_path, generate_img_func, 
                 global_batch_size=64, h=512, w=512, is_train=True):
        
        self.is_train = is_train

        self.ds_static = StaticImageDataset(
            folder_path=static_folder, 
            h=h, w=w,
            is_train=is_train
        )

        self.ds_dynamic = TurboDataset_Img(
            data_path=dynamic_data_path, 
            generate_img_func=generate_img_func,
            h=h, w=w,
            is_train=is_train,
            batch_size=1
        )

        len_static = len(self.ds_static)
        len_dynamic = len(self.ds_dynamic) 
        total_samples = len_static + len_dynamic

        if total_samples == 0:
            raise ValueError("Les répertoires de données fournis sont vides.")

        ratio_dynamic = len_dynamic / total_samples

        self.batch_dynamic = int(round(global_batch_size * ratio_dynamic))
        self.batch_static = global_batch_size - self.batch_dynamic

        self.batch_dynamic = max(1, self.batch_dynamic) if len_dynamic > 0 else 0
        self.batch_static = max(1, self.batch_static) if len_static > 0 else 0

        self.global_batch_size = self.batch_static + self.batch_dynamic
        self.total_steps = total_samples // self.global_batch_size

        mode_str = "ENTRAÎNEMENT" if is_train else "VALIDATION"
        print(f"\n[{mode_str}] --- Configuration MixedDataLoader ---")
        print(f"Dossiers Uniques : {len_static} statiques | {len_dynamic} signaux dynamiques")
        print(f"Composition Lot  : {self.batch_static} statiques + {self.batch_dynamic} dynamiques")

        self.loader_static = DataLoader(
            self.ds_static, 
            batch_size=self.batch_static, 
            shuffle=is_train, 
            drop_last=is_train
        )

        self.ds_dynamic.batch_size = self.batch_dynamic
        self.loader_dynamic = self.ds_dynamic

    def __iter__(self):
        iter_static = iter(self.loader_static)
        iter_dynamic = iter(self.loader_dynamic)

        while True:
            try:
                images_s, targets_s, lens_s = next(iter_static)
                images_d, targets_d, lens_d = next(iter_dynamic)
            except StopIteration:
                break

            images_mixed = torch.cat([images_s, images_d], dim=0)
            targets_mixed = torch.cat([targets_s, targets_d], dim=0)
            lens_mixed = torch.cat([lens_s, lens_d], dim=0)

            if self.is_train:
                indices = torch.randperm(images_mixed.shape[0])
            else:
                indices = torch.arange(images_mixed.shape[0])

            yield images_mixed[indices], targets_mixed[indices], lens_mixed[indices]

    def __len__(self):
        return self.total_steps



def main():
    print("=== DÉMARRAGE DU TEST PIPELINE RECONSTRUCTION (STATIC ID-BASED) ===")

    # Paramètres de test
    H, W = 256, 512 
    GLOBAL_BATCH_SIZE = 16
    # Remplace par tes vrais chemins
    dossier_statique = "../../../../output/final_data/train"
    path_signaux_bruts = "../../../../physionet_data/final_data/train"


    print("\n" + "="*50)
    print("1. TEST DU PIPELINE ENTRAÎNEMENT")
    print("="*50)
    
    train_loader = MixedDataLoader(
        static_folder=dossier_statique,
        dynamic_data_path=path_signaux_bruts, 
        generate_img_func=ecg_image_kit_vectoriel,
        global_batch_size=GLOBAL_BATCH_SIZE,
        h=H, w=W,
        is_train=True
    )

    print(f"[*] Configuration : {train_loader.batch_stat} statiques + {train_loader.batch_dyn} dynamiques par batch.")

    for step, (images, targets, num_windows) in enumerate(train_loader):
        print(f"\n[Batch {step}]")
        print(f"   - Images shape : {images.shape} (Mixte)")
        print(f"   - GT (Signal)  : {targets.shape} -> Reconstruction cible [12, 4000]")

        # Analyse de la provenance (statique vs dynamique) via num_windows ou logs
        if step == 0:
            print("   - Vérification visuelle : Les images de ce batch sont un mélange de rendus CSV et Online.")
            print("   - Augmentation : Les images statiques sont choisies aléatoirement parmi les vues dispos.")
            
        if step >= 1: break

    print("\n" + "="*50)
    print("2. TEST DU PIPELINE VALIDATION")
    print("="*50)
    
    val_loader = MixedDataLoader(
        static_folder=dossier_statique,
        dynamic_data_path=path_signaux_bruts, 
        generate_img_func=ecg_image_kit_vectoriel,
        global_batch_size=GLOBAL_BATCH_SIZE,
        h=H, w=W,
        is_train=False
    )

    for step, (images, targets, num_windows) in enumerate(val_loader):
        print(f"\n[Batch {step}]")
        print(f"   - Images shape : {images.shape}")
        if step == 0:
            print("   - Vérification : En VAL, on utilise UNIQUEMENT le rendu '0001' pour la partie statique.")
            print("   - Reproductibilité : Le batch n'est pas mélangé (shuffle interne off).")
            
        if step >= 1: break

    print("\n=== TEST TERMINÉ AVEC SUCCÈS ===")


if __name__ == "__main__":
    main()
