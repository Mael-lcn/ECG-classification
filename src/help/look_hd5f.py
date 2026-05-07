import os
import argparse
import sys
import math
import numpy as np
from tqdm import tqdm
import h5py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from datetime import datetime



def process_single_file(file_path):
    """
    Worker : Vérifie la shape, puis lit le fichier en 4 chunks pour économiser la RAM.
    """
    file_name = file_path.name
    report_lines = []

    header = f"\n{'='*60}\nFICHIER : {file_name}\n{'='*60}"
    report_lines.append(header)

    try:
        with h5py.File(file_path, 'r') as f:
            if 'tracings' not in f:
                return f"{header}\n[SKIP] La clé 'tracings' est absente du fichier.\n"

            dset = f['tracings']
            shape = dset.shape

            # =========================================================
            # VÉRIFICATION CRITIQUE DES DIMENSIONS (AVANT CHARGEMENT)
            # =========================================================
            if len(shape) != 3:
                return f"{header}\n[CRITIQUE] Shape invalide : attendu 3D (N, C, T), obtenu {shape}.\n"
            
            N, C, T = shape

            if N == 0 or C == 0 or T == 0:
                return f"{header}\n[CRITIQUE] Dimension à 0 détectée : {shape}. Fichier ignoré.\n"
            """
            if C != 12:
                return f"{header}\n[CRITIQUE] Canaux invalides : attendu C=12, obtenu C={C} ({shape}).\n"

            if T <= 100:
                return f"{header}\n[CRITIQUE] Séquence trop courte : attendu T>100, obtenu T={T} ({shape}).\n"
            """
            if not np.issubdtype(dset.dtype, np.number):
                return f"{header}\n[SKIP] Le dataset 'tracings' n'est pas numérique (Type: {dset.dtype}).\n"

            # Initialisation du rapport de base
            dset_report = [f"\n   [TARGET] 'tracings' | Shape: {shape} | Type: {dset.dtype}"]

            # =========================================================
            # LECTURE PAR CHUNKS (COUPÉ EN 4)
            # =========================================================
            stats = {
                "has_nan": False,
                "has_inf": False,
                "has_underflow": False,
                "min_val": float('inf'),
                "max_val": float('-inf')
            }

            # On divise N en 4 parts égales (arrondi au supérieur)
            chunk_size = math.ceil(N / 4)

            # Boucle sur les 4 morceaux
            for i in range(4):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, N)

                # Sécurité si N est très petit (ex: N < 4)
                if start_idx >= N:
                    break

                # Chargement d'un quart des données en RAM
                chunk_data = dset[start_idx:end_idx, :, :]

                # 1. Mise à jour des corruptions (NaN / Inf)
                if not stats["has_nan"] and np.isnan(chunk_data).any():
                    stats["has_nan"] = True
                if not stats["has_inf"] and np.isinf(chunk_data).any():
                    stats["has_inf"] = True

                # 2. Mise à jour Underflow (FP16)
                if not stats["has_underflow"] and np.any((np.abs(chunk_data) > 0) & (np.abs(chunk_data) < 6.1e-5)):
                    stats["has_underflow"] = True

                # 3. Mise à jour des bornes Min / Max
                try:
                    c_min = float(np.nanmin(chunk_data))
                    c_max = float(np.nanmax(chunk_data))
                    if c_min < stats["min_val"]:
                        stats["min_val"] = c_min
                    if c_max > stats["max_val"]:
                        stats["max_val"] = c_max
                except ValueError:
                    # Passe ici si ce chunk entier ne contient QUE des NaN
                    pass 

            # --- RÉDACTION DU RAPPORT FINAL ---
            if stats["has_nan"]:
                dset_report.append("      >>> ALERTE : Contient des NaN (Corrompu)")
            if stats["has_inf"]:
                dset_report.append("      >>> ALERTE : Contient des valeurs Infinies (Inf)")
            if stats["has_underflow"]:
                dset_report.append("      >>> WARNING : Underflow détecté (valeurs positives < 6.1e-5)")

            if stats["min_val"] != float('inf'):
                dset_report.append(f"      -> Min: {stats['min_val']:.6f} | Max: {stats['max_val']:.6f}")
                # Check Overflow Float16
                if abs(stats['max_val']) > 65500 or abs(stats['min_val']) > 65500:
                    dset_report.append("      -> WARNING : Valeurs > 65500 (Risque Overflow Float16)")
            else:
                dset_report.append("      -> Dataset contenant uniquement des NaN.")

            report_lines.extend(dset_report)

    except Exception as e:
        error_details = traceback.format_exc()
        return f"{header}\n[ERREUR FATALE] Échec lors du traitement :\n{error_details}\n"

    return "\n".join(report_lines)


def main():
    """
    Point d'entrée principal.
    """
    parser = argparse.ArgumentParser(description="Inspecteur HDF5 'tracings' (Chunk Load)")
    parser.add_argument('-i', '--input', type=str, default='../output/normalize_data/', help="Dossier contenant les fichiers .hdf5")
    parser.add_argument('--log', type=str, default='../output/audit_report.log', help="Chemin du fichier de log")

    default_workers = max(1, os.cpu_count()-1)
    parser.add_argument('--workers', type=int, default=default_workers, help=f"Processus parallèles (défaut: {default_workers})")

    args = parser.parse_args()
    input_dir = Path(args.input)

    if not input_dir.exists():
        print(f"[ERREUR] Dossier introuvable : {input_dir}")
        sys.exit(1)

    files = list(input_dir.glob("*.hdf5"))
    if not files:
        print("[INFO] Aucun fichier .hdf5 trouvé.")
        sys.exit(0)

    with open(args.log, 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== AUDIT HDF5 DÉMARRÉ LE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_file.write(f"[INFO] Analyse de {len(files)} fichiers (Cible: 'tracings').\n")
        log_file.write(f"[INFO] Mode : RAM protégée (Fichiers coupés en 4 chunks).\n")
        log_file.write(f"[INFO] Workers actifs : {args.workers}\n\n")

        print(f"Analyse en cours (Chunk mode)... Résultats redirigés dans : {args.log}")

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {executor.submit(process_single_file, f): f for f in files}

            with tqdm(total=len(files), desc="Progression", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        report = future.result()
                        log_file.write(report + "\n")
                        log_file.flush() 
                    except Exception as exc:
                        error_details = traceback.format_exc()
                        log_file.write(f"\n{'='*60}\nFICHIER : {file_path.name}\n{'='*60}\n")
                        log_file.write(f"[EXCEPTION MAIN] Erreur critique du worker :\n{error_details}\n")
                        log_file.flush()
                    finally:
                        pbar.update(1)

        fin_msg = f"\n{'='*80}\n[TERMINE] Audit complet effectué le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n"
        log_file.write(fin_msg)

    print("\nTerminé ! Tu peux consulter le fichier de log.")


if __name__ == "__main__":
    main()
