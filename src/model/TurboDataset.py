import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

from src.dataset.normalization import TARGET_FREQ



# Constantes temporelles
MAX_TEMPS = 144
MAX_SIGNAL_LENGTH = MAX_TEMPS * TARGET_FREQ + 10


class TurboDataset(IterableDataset):
    """
    Dataset itérable de niveau industriel.
    
    Stratégie : 'Global Shuffle, Local Fuzzy Sort'.
    - I/O : Lecture de blocs contigus (Mega-Batches) pour saturer la bande passante disque.
    - Padding : Tri local avec 20% de bruit pour minimiser le vide sans biaiser le gradient.
    - CPU : Offloading de la création du masque vers le GPU en renvoyant uniquement les longueurs.
    """
    def __init__(self, data_path, batch_size=64, mega_batch_size=8, 
                 use_static_padding=False, max_signal_length=MAX_SIGNAL_LENGTH,
                 is_train=True):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.mega_batch_size = mega_batch_size
        self.use_static_padding = use_static_padding
        self.max_signal_length = max_signal_length
        self.is_train = is_train

        # Identification des fichiers shards
        self.shard_files = sorted(glob.glob(os.path.join(data_path, "*_signals.npy")))
        if not self.shard_files:
            raise FileNotFoundError(f"Aucun fichier shard détecté dans {data_path}")

        # Calcul de la volumétrie totale via mmap pour économiser la RAM
        self.total_samples = 0
        for f in self.shard_files:
            lab_f = f.replace("_signals.npy", "_labels.npy")
            self.total_samples += np.load(lab_f, mmap_mode='r').shape[0]

    def __iter__(self):
        worker_info = get_worker_info()
        indices_shards = list(range(len(self.shard_files)))

        # Répartition entre les workers multiprocessing
        if worker_info is not None:
            indices_shards = indices_shards[worker_info.id :: worker_info.num_workers]

        # Mélange global des fichiers
        if self.is_train:
            np.random.shuffle(indices_shards)

        for shard_idx in indices_shards:
            sig_f = self.shard_files[shard_idx]
            lab_f = sig_f.replace("_signals.npy", "_labels.npy")
            met_f = sig_f.replace("_signals.npy", "_meta.csv")

            # Ouverture des descripteurs sans chargement immédiat
            sig_mmap = np.load(sig_f, mmap_mode='r')
            lab_all = np.load(lab_f)
            # Lecture optimisée des métadonnées via le moteur C de Pandas
            len_all = pd.read_csv(met_f, usecols=['length'], engine='c')['length'].values

            shard_len = len(sig_mmap)
            # Découpage en Mega-Batches pour l'efficacité I/O
            mb_starts = list(range(0, shard_len, self.mega_batch_size))

            if self.is_train:
                np.random.shuffle(mb_starts)

            for start in mb_starts:
                end = min(start + self.mega_batch_size, shard_len)

                # 1. Lecture contiguë forcée
                mb_signals = np.array(sig_mmap[start:end]) 
                mb_labels = lab_all[start:end]
                mb_lengths = len_all[start:end]
                mb_size = len(mb_signals)

                # 2. Tri Bruité
                # Groupe les tailles proches pour réduire le padding tout en gardant du chaos sain
                if self.is_train:
                    noise = np.random.uniform(-0.2, 0.2, size=mb_lengths.shape)
                    sort_keys = mb_lengths * (1 + noise)
                else:
                    sort_keys = mb_lengths  # Déterministe en validation

                sort_idx = np.argsort(sort_keys)
                mb_signals = mb_signals[sort_idx]
                mb_labels = mb_labels[sort_idx]
                mb_lengths = mb_lengths[sort_idx]

                # 3. Mélange des mini-batchs au sein du Mega-Batch
                # Empêche le modèle de voir une progression monotone des tailles
                batch_indices = list(range(0, mb_size, self.batch_size))
                if self.is_train:
                    np.random.shuffle(batch_indices)

                for j in batch_indices:
                    idx_end = min(j + self.batch_size, mb_size)

                    b_sig = mb_signals[j:idx_end]
                    b_lab = mb_labels[j:idx_end]
                    b_len = mb_lengths[j:idx_end]

                    cur_bs = len(b_sig)
                    if cur_bs <= 1: continue

                    if b_len.max() < 50:
                        # On calcule l'offset global dans le fichier pour savoir de quel échantillon on parle
                        """
                        global_idx_start = start + j
                        print(f"\n[CRITICAL DATA ERROR]")
                        print(f"  - Fichier Shard : {os.path.basename(sig_f)}")
                        print(f"  - Range indices dans shard : [{global_idx_start} : {global_idx_start + cur_bs}]")
                        print(f"  - Longueurs trouvées dans le batch : {b_len.tolist()}")
                        print(f"  - Statut : BATCH IGNORÉ (Trop court pour l'architecture CNN)")
                        print("-" * 30)
                        """
                        continue

                    # Stratégie de padding dynamique optimisée par le tri
                    target_t = self.max_signal_length if self.use_static_padding else int(b_len.max())

                    # Allocation directe sans initialisation
                    batch_x = torch.empty((cur_bs, 12, target_t), dtype=torch.float32)

                    # Remplissage par copie mémoire et padding ciblé
                    for k in range(cur_bs):
                        l = b_len[k]
                        batch_x[k, :, :l] = torch.from_numpy(b_sig[k, :, :l])
                        if l < target_t:
                            batch_x[k, :, l:] = 0.0

                    batch_y = torch.from_numpy(b_lab)
                    batch_lens = torch.from_numpy(b_len).long()

                    # On renvoie les longueurs pour que le GPU génère le masque
                    yield batch_x, batch_y, batch_lens

    def __len__(self):
        """Estimation du nombre total de mini-batchs."""
        return int(np.ceil(self.total_samples / self.batch_size))
