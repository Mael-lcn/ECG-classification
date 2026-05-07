import argparse
import numpy as np
import pandas as pd
import h5py
import glob
import sys
import os
import time
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.dataset.aux import re_sampling, z_norm, get_active_boundaries



# Config GPU
def get_device():
    """Détecte et configure le device (MPS pour Mac, CUDA pour Nvidia, sinon CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"Accélération matérielle activée sur : {DEVICE}")


def to_optimized_tensor(numpy_array):
    """
    Convertit un array Numpy (N, T, C) en Tenseur PyTorch (N, C, T) optimisé pour le GPU.

    1. Conversion en Tensor.
    2. Permutation des axes : (Batch, Time, Channel) -> (Batch, Channel, Time).
    3. Contiguous() pour l'alignement mémoire (évite les bugs View/Reshape).
    4. Transfert vers le Device.
    
    Args:
        numpy_array (np.ndarray): Données brutes au format (N, T, C).
        
    Returns:
        torch.Tensor: Tenseur (N, C, T) sur le DEVICE.
    """
    tensor = torch.from_numpy(numpy_array)
    
    # Permutation (N, T, C) -> (N, C, T)
    tensor = tensor.permute(0, 2, 1)

    # Alignement mémoire obligatoire après permute
    tensor = tensor.contiguous()

    # Assurance précision float32
    if tensor.dtype == torch.float64:
        tensor = tensor.float()

    return tensor.to(DEVICE)


def generate_strict_physio_batch(n_samples_per_freq=10, duration_sec=2.0):
    """
    Génère un batch de données synthétiques directement au bon format.
    
    Returns:
        dict: {'tracings': Tensor(N, C, T), 'exam_id': list}
        pd.DataFrame: Métadonnées
    """
    target_fs_list = [1000, 257, 250, 500, 128] 
    max_fs = max(target_fs_list)
    buffer_len = int(duration_sec * max_fs) + 200

    total_samples = len(target_fs_list) * n_samples_per_freq
    # Numpy travaille en (N, T, C) par défaut
    tracings_np = np.zeros((total_samples, buffer_len, 12), dtype=np.float32)
    exam_ids = []
    meta_rows = []

    current_idx = 0
    print(f"\n[SYNTHETIC] Génération de {total_samples} examens de stress...")

    for fs in target_fs_list:
        n_points = int(duration_sec * fs)
        t = np.linspace(0, duration_sec, n_points).reshape(-1, 1)

        for i in range(n_samples_per_freq):
            e_id = f"SYNTH_Fs{fs}_{i}"
            b_id = e_id.encode('utf-8')

            # Paramètres aléatoires
            heart_rate_hz = np.random.uniform(0.5, 3.0) 
            amps = np.random.uniform(-40.0, 40.0, size=(1, 12))
            phases = np.random.uniform(0, 2*np.pi, size=(1, 12))
            offsets = np.random.uniform(-100, 100, size=(1, 12))

            # Synthèse signal
            clean_sig = amps * np.sin(2 * np.pi * heart_rate_hz * t + phases) + offsets
            noise = np.random.normal(0, 0.1 * np.abs(amps), (n_points, 12))

            # Remplissage Numpy (Time, Channel)
            tracings_np[current_idx, :n_points, :] = clean_sig + noise

            exam_ids.append(b_id)
            meta_rows.append({'exam_id': e_id, 'frequences': fs})
            current_idx += 1

    # Shuffle pour éviter les biais d'ordre
    perm = np.random.permutation(total_samples)

    print(f"   -> Conversion en Tenseur (N, C, T) sur {DEVICE}...")
    tracings_gpu = to_optimized_tensor(tracings_np[perm])

    return {
        'tracings': tracings_gpu, 
        'exam_id': [exam_ids[i] for i in perm]
    }, pd.DataFrame(meta_rows)


# TODO mouais bof à revoir
def load_real_data(input_dir, n_samples=50):
    """
    Charge des données réelles et les retourne au format Tensor (N, C, T).
    """
    print(f"\n[LOADING] Scan du dossier : {input_dir}")
    h5_files = sorted(glob.glob(os.path.join(input_dir, "*.hdf5")))
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    if not h5_files:
        raise FileNotFoundError("Aucun fichier .hdf5 trouvé.")

    print("   -> Chargement des métadonnées (CSV)...")
    if csv_files:
        try:
            df_global = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            df_global = df_global.set_index('exam_id')
        except Exception as e:
            raise ValueError(f"Erreur CSV : {e}")
    else:
        raise FileNotFoundError("Aucun CSV trouvé.")

    collected_tracings = []
    collected_ids = []
    collected_meta_rows = []

    print(f"   -> Extraction HDF5 ({n_samples} samples/file)...")

    for i, h5_path in enumerate(h5_files):
        try:
            with h5py.File(h5_path, 'r') as f:
                # Chargement partiel pour économiser la RAM
                dset_tracings = f['tracings']
                dset_ids = f['exam_id']
                n_avail = dset_tracings.shape[0]
                n_to_pick = min(n_samples, n_avail)

                # Indices aléatoires
                indices = np.sort(np.random.choice(n_avail, n_to_pick, replace=False))

                # Lecture directe
                # Note: h5py retourne du Numpy (N, T, C)
                signals = dset_tracings[indices] 
                ids = dset_ids[indices]

                pass

        except Exception as e:
            print(e)


def test_hard_cases_boundaries():
    print("\n" + "="*60)
    print(f"TEST UNITAIRE : src.dataset.aux.get_active_boundaries")
    print(f"Format : (N, C, T) sur {DEVICE}")
    print("="*60)
    
    # --- CAS 1 : Le "Décalage Extrême" (Le test du rectangle) ---
    print("1. Test Décalage Canaux (Le 'Rectangle')...")

    # 1 Patient, 3 Canaux, 10 Pas de temps
    t1 = torch.zeros(1, 3, 10, device=DEVICE)

    # C0 : Actif au début [0, 2] -> t1[0, 0, 0:3]
    t1[0, 0, 0:3] = 1.0 
    # C1 : Actif au milieu [4, 6] -> t1[0, 1, 4:7]
    t1[0, 1, 4:7] = 1.0
    # C2 : Actif à la fin [8, 9]  -> t1[0, 2, 8:10]
    t1[0, 2, 8:10] = 1.0

    s, e = get_active_boundaries(t1)

    print(f"   Input (N,C,T): C0[0-2], C1[4-6], C2[8-10] sur T=10")
    print(f"   Attendu : Start=0, End=10")
    print(f"   Obtenu  : Start={s.item()}, End={e.item()}")

    if s.item() == 0 and e.item() == 10:
        print("   -> SUCCÈS")
    else:
        print("   -> ÉCHEC")

    # --- CAS 2 : Le signal "Inclus" ---
    print("\n2. Test Signal Inclus...")
    t2 = torch.zeros(1, 2, 20, device=DEVICE) # (1, 2, 20)
    t2[0, 0, 5:15] = 1.0 # Le large
    t2[0, 1, 8:10] = 1.0 # Le court

    s, e = get_active_boundaries(t2)
    print(f"   Attendu : Start=5, End=15")
    print(f"   Obtenu  : Start={s.item()}, End={e.item()}")
    assert s.item() == 5 and e.item() == 15, "Erreur logique inclus"
    print("   -> SUCCÈS")

    # --- CAS 3 : Le Patient Vide ---
    print("\n3. Test Signal Vide (Flatline)...")
    t3 = torch.zeros(1, 12, 50, device=DEVICE)
    s, e = get_active_boundaries(t3)
    print(f"   Attendu : Start=0, End=0")
    print(f"   Obtenu  : Start={s.item()}, End={e.item()}")
    assert s.item() == 0 and e.item() == 0, "Le vide doit retourner 0,0"
    print("   -> SUCCÈS")

    print("\n" + "="*60)


def test_bilateral_padding_logic():
    """
    Test unitaire dédié à la fonction add_bilateral_padding.
    Vérifie :
    1. Que le signal court est bien centré dans la cible (Padding).
    2. Que les dimensions sont respectées.
    3. Que le code plante (RuntimeError) si le signal est trop long (comportement voulu).
    """
    from src.dataset.aux import add_bilateral_padding  # Assurez-vous que l'import est là

    print("\n" + "="*60)
    print("TEST UNITAIRE : Bilateral Padding (Centrage)")
    print("="*60)

    target_size = 10
    
    # --- Création d'un Batch Synthétique (N=3, C=1, T=20) ---
    # On met T=20 pour avoir de la marge autour des signaux
    input_tensor = torch.zeros((3, 1, 20), dtype=torch.float32, device=DEVICE)

    # PATIENT 0 : Signal pair (Longueur 4) -> Doit être centré
    # Indices [5:9] = 1.0
    input_tensor[0, :, 5:9] = 1.0
    
    # PATIENT 1 : Signal impair (Longueur 5) -> Doit être centré (décalage asymétrique de 1)
    # Indices [2:7] = 2.0
    input_tensor[1, :, 2:7] = 2.0

    # PATIENT 2 : Signal TROP LONG (Longueur 15) -> Doit planter
    # Indices [0:15] = 3.0
    input_tensor[2, :, 0:15] = 3.0

    print(f"   Configuration : Target Size = {target_size}")
    print(f"   Input Batch (N=3) :")
    print(f"     - P0: Len 4 (Pair)")
    print(f"     - P1: Len 5 (Impair)")
    print(f"     - P2: Len 15 (Trop long -> Doit Crash)")

    # --- TEST 1 : Padding Valide (P0 et P1 seulement) ---
    print("\n1. Test du Padding (Cas valides)...")

    # On prend uniquement les 2 premiers patients valides
    valid_batch = input_tensor[0:2]
    padded_output = add_bilateral_padding(valid_batch, target_size=target_size)

    # Vérification P0 (Len 4 dans Target 10 -> Offset (10-4)//2 = 3)
    # Attendu : [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    row0 = padded_output[0, 0].cpu().numpy()
    check0 = np.allclose(row0[3:7], 1.0) and (row0[:3].sum() == 0) and (row0[7:].sum() == 0)

    print(f"   [P0] Attendu : 3 zéros - 4 ones - 3 zéros")
    print(f"        Obtenu  : {row0}")
    print(f"        Statut  : {'SUCCÈS' if check0 else 'ÉCHEC'}")

    # Vérification P1 (Len 5 dans Target 10 -> Offset (10-5)//2 = 2)
    # Attendu : [0, 0, 2, 2, 2, 2, 2, 0, 0, 0]
    row1 = padded_output[1, 0].cpu().numpy()
    check1 = np.allclose(row1[2:7], 2.0) and (row1[:2].sum() == 0) and (row1[7:].sum() == 0)

    print(f"   [P1] Attendu : 2 zéros - 5 twos - 3 zéros")
    print(f"        Obtenu  : {row1}")
    print(f"        Statut  : {'SUCCÈS' if check1 else 'ÉCHEC'}")

    # --- TEST 2 : Crash Attendu (P2) ---
    print("\n2. Test du Crash de Sécurité (Signal > Target)...")
    
    try:
        # On passe le patient 2 qui fait 15 de long (target=10)
        # Cela DOIT lever une RuntimeError lors de l'assignation slice
        _ = add_bilateral_padding(input_tensor[2:3], target_size=target_size)
        print("   ÉCHEC : La fonction aurait dû planter mais a continué.")
    except RuntimeError as e:
        print(f"   SUCCÈS : La fonction a planté comme prévu.")
        print(f"      Message : {e}")
    except Exception as e:
        print(f"   SUCCÈS PARTIEL : A planté, mais pas l'erreur attendue ({type(e)}).")

    print("\n" + "="*60)


def run_checks(data, csv_meta, target_fo=400):
    print("="*60)
    print("DÉBUT DU PIPELINE DE VALIDATION")
    print("="*60)
    start_time = time.time()

    test_hard_cases_boundaries()
    #test_bilateral_padding_logic()

    # Vérification du format d'entrée
    tracings = data['tracings']
    print(f"\n[CHECK INPUT] Shape: {tracings.shape}, Device: {tracings.device}")
    print(f"              Contiguous: {tracings.is_contiguous()}")

    if tracings.shape[1] > tracings.shape[2]:
        print("ALERTE : Il semble que le format soit (N, T, C) au lieu de (N, C, T).")
        print("Cela risque de causer des erreurs dans re_sampling.")

    # 2. RESAMPLING
    print(f"\n1. Exécution re_sampling (Cible: {target_fo}Hz)...")

    resampled = re_sampling(data, csv_meta, fo=target_fo)

    print(f"   Succès. Output Shape: {resampled.shape} (N, C, T)")

    # Check intégrité sur GPU
    if not torch.isfinite(resampled).all():
        print("   ALERTE : NaNs ou Infs détectés après resampling !")
    else:
        print("   Intégrité basique OK.")

    # 3. Z-NORM
    print(f"\n2. Exécution z_norm...")
    try:
        normalized = z_norm(resampled)
        print(f"   Succès. Output Shape: {normalized.shape}")
    except Exception as e:
        print(f"   CRASH z_norm: {e}")
        return

    # 4. audit statistique
    print(f"\n3. Audit Statistique (Format N, C, T)...")

    # Calcul des bornes sur GPU
    starts, ends = get_active_boundaries(normalized)

    # Retour CPU pour affichage et calcul stats Numpy
    normalized_np = normalized.cpu().numpy() # (N, C, T)
    starts_np = starts.cpu().numpy()
    ends_np = ends.cpu().numpy()

    n_samples = normalized.shape[0]
    stats_ok_count = 0
    samples_to_show = min(n_samples, 20)

    print(f"   {'IDX':<6} | {'MEAN (Max)':<12} | {'STD (Range)':<20} | {'PADDING'}")
    print("   " + "-"*55)

    for i in range(n_samples):
        # sig est maintenant (C, T) car on a permuté !
        sig = normalized_np[i] 
        s_idx, e_idx = starts_np[i], ends_np[i]

        if e_idx == 0:
            if i < samples_to_show:
                print(f"   {i:<6} | {'VIDE':<12} | {'VIDE':<20} | OK")
            continue

        # Slicing sur l'axe T (dim 1 dans Numpy (C, T))
        active = sig[:, s_idx:e_idx] 
        padding_after = sig[:, e_idx:]

        # Stats sur l'axe T (axis=1)
        means = np.mean(active, axis=1)
        stds = np.std(active, axis=1)

        max_mean_dev = np.max(np.abs(means))
        min_std, max_std = np.min(stds), np.max(stds)

        # Check Padding
        pad_clean = True
        if padding_after.size > 0:
            pad_max = np.max(np.abs(padding_after))
            if pad_max > 1e-5: pad_clean = False

        # Validité (Mean < 0.2, 0.5 < Std < 1.5)
        is_stat_valid = (max_mean_dev < 0.2) and (0.5 < min_std) and (max_std < 1.5)
        if is_stat_valid: stats_ok_count += 1
        
        if i < samples_to_show:
            stat_str = "OK" if is_stat_valid else "BAD"
            pad_str = "OK" if pad_clean else "DIRTY"
            print(f"   {i:<6} | {max_mean_dev:<12.4f} | {min_std:.2f}-{max_std:.2f} ({stat_str})   | {pad_str}")

    print("   " + "-"*55)
    print(f"\n[RAPPORT FINAL] Stats Valides: {stats_ok_count}/{n_samples} ({(stats_ok_count/n_samples)*100:.1f}%)")
    print(f"Temps d'exécution : {time.time() - start_time:.2f}s")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default="../output/normalize_data/")
    parser.add_argument('-a', '--use_input', action=argparse.BooleanOptionalAction)
    parser.add_argument('-n', '--n_samples', type=int, default=500)
    parser.add_argument('-f', '--freq_out', type=int, default=400)
    args = parser.parse_args()

    if args.use_input:
        if not os.path.exists(args.input_dir):
            sys.exit(f"ERREUR: Dossier introuvable {args.input_dir}")
        data_dict, csv_df = load_real_data(args.input_dir, n_samples=args.n_samples)
        run_checks(data_dict, csv_df, target_fo=args.freq_out)
    else:
        per_freq = max(1, args.n_samples // 5) 
        data_dict, csv_df = generate_strict_physio_batch(n_samples_per_freq=per_freq)
        run_checks(data_dict, csv_df, target_fo=args.freq_out)


if __name__ == "__main__":
    main()
