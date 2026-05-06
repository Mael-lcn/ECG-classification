import os, sys
from pathlib import Path
import argparse
import time
import gc

import torch
import random
import threading
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import ThreadPool

root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

# 2. On fait un import absolu depuis la racine
from src.dataset.aux import *


# --- CONFIGURATION GLOBALE ---
TARGET_FREQ = 400


# Limite de sécurité VRAM (en Go). Le GPU fait environ 9.64, on réserve 600Mo pour le système/overhead
VRAM_LIMIT_GB = 8.64
current_vram_usage = 0.0
vram_lock = threading.Condition()

# Contient les noyaux de resample partagé
global_resamplers = {}
resamplers_lock = threading.Lock()


# Suivi du record de taille
global_stats = {
    'max_t': 0,
    'file_origin': "",
    'exam_id': "",
    'freq': 0
}
stats_lock = threading.Lock()


def estimate_vram_gb(path_h5):
    """
    Estime la mémoire nécessaire pour un file.
    """
    try:
        with h5py.File(path_h5, 'r') as f:
            shape = f['tracings'].shape
            # N * C * T * 4 bytes / 1024^3
            size_gb = (torch.tensor(shape).prod().item() * 4) / (1024**3)
            # Facteur multiplicatif car resampling + z-norm créent des copies temporaires
            return max(size_gb * 1, 0.2)
    except:
        return 0.5


# TODO Freq n'est pas maj et rename de normal_ecg -> SNR non plus (meme si je ne suis pas sur qu'il soit pertinent)
def unified_worker(task, output):
    """
    Worker optimisé pour le traitement ECG haute performance sous contrainte VRAM.

    Stratégie :
    1. Streaming GPU : Charge et traite par petits blocs pour minimiser l'empreinte VRAM.
    2. Déchargement Immédiat : Renvoie les tenseurs sur RAM (CPU) dès la fin du calcul.
    3. Assemblage CPU: Utilise une pré-allocation et une consommation inversée de la liste 
       pour éviter les réallocations et copies mémoire lors de la fusion finale.

    Args:
        task (tuple): (mode, name, (path_h5, path_csv)).
        output (str): Dossier de destination.
    """
    global current_vram_usage
    mode, name, path = task
    path_h5, path_csv = path

    # Admission et verouillage VRAM
    cost = estimate_vram_gb(path_h5)

    with vram_lock:
        # Si on a assez de VRAM ou que aucun file n'est traité, on autorise le traitement
        while (current_vram_usage + cost > VRAM_LIMIT_GB) and (current_vram_usage > 0.1):
            vram_lock.wait()

        current_vram_usage += cost


    # Structures de données
    dataset = {}
    csv_full = None
    processed_chunks_cpu = []
    all_lengths = []
    all_starts = []

    try:
        # Sélection du device
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Lecture des méta données sur CPU
        exam_ids, csv_full = load_metadata(path_h5, path_csv)
        N = len(exam_ids)

        # Indexation optimisée
        csv_full['exam_id'] = csv_full['exam_id'].astype(str)
        csv_indexed = csv_full.set_index('exam_id')

        # On divise le fichier en morceaux égaux
        fragment_factor = 8
        chunks_boundaries = [
            ((i * N) // fragment_factor, ((i + 1) * N) // fragment_factor) 
            for i in range(fragment_factor)
        ]

        processed_chunks_cpu = []
        all_lengths = []
        all_starts = []
        all_valid_ids = []

        # ==========================================================
        # BOUCLE DE TRAITEMENT PAR CHUNKS
        # ==========================================================
        for i, (start, end) in enumerate(chunks_boundaries):
            if start == end: continue

            # A. Chargement (Disque -> GPU)
            chunk_gpu = load_chunk(path_h5, start, end, device)
            chunk_ids_raw = exam_ids[start:end]

            # Extraction subset CSV uniquement
            chunk_ids_str = [eid.decode('utf-8') if isinstance(eid, bytes) else str(eid) for eid in chunk_ids_raw]
            chunk_csv = csv_indexed.reindex(chunk_ids_str).reset_index()

            # Supprime les NANs
            clean_data, chunk_csv = remove_nan_records(
                {'tracings': chunk_gpu, 'exam_id': chunk_ids_raw}, 
                chunk_csv
            )

            # Mise à jour des tenseurs et IDs locaux avec les données saines
            chunk_gpu = clean_data['tracings']
            chunk_ids_raw = clean_data['exam_id']

            # Si tout le chunk était corrompu (0 signal restant), on passe au suivant
            if chunk_gpu.shape[0] == 0:
                print(f"   -> [{name}] {i+1}/{fragment_factor} ignoré (100% corrompu).", flush=True)
                continue

            # On sauvegarde les IDs valides pour l'assemblage final
            all_valid_ids.append(chunk_ids_raw)

            # D. Calcul Métadonnées
            chunk_gpu, end_idxs = clean_values(chunk_gpu)

            # B. Traitement (Resampling / Filtres)
            if mode == 'D1':
                # Le resampling crée un nouveau tenseur. On lui passe clean_data.
                chunk_gpu = re_sampling(clean_data, chunk_csv, end_idxs, global_resamplers, resamplers_lock, fo=TARGET_FREQ)

            # Nettoyage immédiat
            del clean_data, chunk_csv, chunk_ids_str

            # C. Normalisation (In-Place pour économiser VRAM)
            chunk_gpu = z_norm(chunk_gpu)

            # D. Calcul Métadonnées
            s_idx, e_idx = get_active_boundaries(chunk_gpu, threshold=1e-5)

            # Transfert métadonnées vers CPU (Numpy list)
            all_lengths.append((e_idx - s_idx).cpu().numpy())
            all_starts.append(s_idx.cpu().numpy())

            # E. Décharge (GPU -> CPU)
            chunk_cpu = chunk_gpu.cpu()
            processed_chunks_cpu.append(chunk_cpu)

            # F. Nettoyage GPU
            del chunk_gpu, s_idx, e_idx
            print(f"   -> [{name}] {i+1}/{fragment_factor} processed.", flush=True)

        # ==========================================================
        # ASSEMBLAGE FINAL SUR CPU
        # ==========================================================
        # Sécurité absolue : si le fichier entier était corrompu, on l'ignore
        if not processed_chunks_cpu:
            print(f"   [SKIP] Fichier {name} entièrement ignoré (100% corrompu).")
            return

        # Calcul des dimensions finales basées sur les données qui ont survécu
        total_rows = sum(c.shape[0] for c in processed_chunks_cpu)
        max_t = max(c.shape[2] for c in processed_chunks_cpu)
        channels = processed_chunks_cpu[0].shape[1]

        # Allocation avec Padding Implicite
        dataset['tracings'] = torch.zeros((total_rows, channels, max_t), 
                                          dtype=torch.float32, 
                                          device='cpu')

        # Remplissage
        processed_chunks_cpu.reverse()

        current_idx = 0
        while processed_chunks_cpu:
            chunk = processed_chunks_cpu.pop()
            n, _, t = chunk.shape

            # Copie RAM -> RAM
            dataset['tracings'][current_idx : current_idx + n, :, :t] = chunk
            current_idx += n

            del chunk

        # On concatène que les IDs qui ont passé le filtre
        final_valid_exam_ids = np.concatenate(all_valid_ids)
        dataset['exam_id'] = final_valid_exam_ids

        # Reconstruction du csv aligné
        final_h5_ids = [eid.decode('utf-8') if isinstance(eid, bytes) else str(eid) for eid in final_valid_exam_ids]
        final_csv = csv_indexed.reindex(final_h5_ids).reset_index()

        # Injection rapide via Numpy
        final_csv['length'] = np.concatenate(all_lengths)
        final_csv['start_offset'] = np.concatenate(all_starts)

        if mode != 'D1':
            final_csv.rename(columns={"normal_ecg": "NSR"}, inplace=True)

        current_lengths = np.concatenate(all_lengths)
        local_max_idx = np.argmax(current_lengths)
        local_max_val = current_lengths[local_max_idx]

        with stats_lock:
            if local_max_val > global_stats['max_t']:
                global_stats['max_t'] = local_max_val
                global_stats['file_origin'] = name

                # Récupération de l'ID du record
                target_id = final_h5_ids[local_max_idx]
                global_stats['exam_id'] = target_id

                try:
                    # On filtre csv_full pour trouver la fréquence associée à cet ID
                    record_row = csv_full[csv_full['exam_id'].astype(str) == str(target_id)]
                    if not record_row.empty:
                        # On prend la première valeur trouvée dans la colonne 'frequences'
                        global_stats['freq'] = record_row['frequences'].values[0]
                    else:
                        global_stats['freq'] = "ID non trouvé"
                except Exception:
                    global_stats['freq'] = "Erreur lecture"

        # Save
        write_results(dataset, final_csv, name, output)

    except Exception as e:
        print(f"   [ERREUR FATALE] {name} : {str(e)}")
        raise e

    finally:
        # Nettoyage final
        # Suppression des références aux gros objets
        if 'dataset' in locals(): del dataset
        if 'processed_chunks_cpu' in locals(): del processed_chunks_cpu
        if csv_full is not None: del csv_full

        # Appel explicite au GC Python
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Libération du slot VRAM dans le gestionnaire global
        with vram_lock:
            current_vram_usage -= cost
            vram_lock.notify_all()



def run(args):
    """
    Orchestre le traitement global en mélangeant les datasets pour une efficacité maximale.
    """
    start_time = time.time()
    os.makedirs(args.output, exist_ok=True)

    # Collecte et étiquetage des tâches
    print("Indexation des fichiers...")
    all_tasks = []

    d1_files = collect_files(args.dataset1)
    for name, path in d1_files.items():
        all_tasks.append(('D1', name, path))

    d2_files = collect_files(args.dataset2)
    for name, path in d2_files.items():
        all_tasks.append(('D2', name, path))

    if not all_tasks:
        print("Aucun fichier trouvé.")
        return

    # 2. Mélange aléatoire
    # Permet de lisser la charge GPU en mixant gros fichiers (D1) et petits (D2)
    random.shuffle(all_tasks)

    print(f"Traitement de {len(all_tasks)} fichiers avec {args.workers} threads.")
    print(f"Gestion dynamique de la VRAM (Limite: {VRAM_LIMIT_GB} GB).")

    # Exécution parallèle
    worker_func = partial(unified_worker, output=args.output)

    with ThreadPool(args.workers) as pool:
        list(tqdm(pool.imap_unordered(worker_func, all_tasks), 
                  total=len(all_tasks), 
                  desc='Global Preprocessing'))

    if global_stats['max_t'] > 0:
        print("\n" + "="*60)
        print("RECORD DU PLUS GRAND SIGNAL")
        print("-"*60)
        print(f"ID Examen      : {global_stats['exam_id']}")
        print(f"Fichier Source : {global_stats['file_origin']}")
        print(f"Taille brute   : {global_stats['max_t']} points")
        print(f"Fréq. d'origine: {global_stats['freq']} Hz")
        print(f"Durée réelle   : {global_stats['max_t'] / TARGET_FREQ:.2f} secondes (à {TARGET_FREQ}Hz)")
        print("="*60 + "\n")

    elapsed = time.time() - start_time
    print(f"\nPipeline terminée en {int(elapsed // 60)}m {int(elapsed % 60)}s.")



def main():
    parser = argparse.ArgumentParser(description="Pipeline ECG Mixte avec Budget VRAM")
    parser.add_argument('-d1', '--dataset1', type=str, default='../../../../output/dataset1/')
    parser.add_argument('-d2', '--dataset2', type=str, default='../../../../data/15_prct/')
    parser.add_argument('-o', '--output', type=str, default='../../../../output/normalize_data')
    parser.add_argument('-w', '--workers', type=int, default=8, help="Nb de threads (I/O)")

    args = parser.parse_args()

    # Config PyTorch pour éviter la fragmentation CUDA
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    run(args)


if __name__ == '__main__':
    main()
