import os, glob
import sys
import json
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import wandb

from sklearn.metrics import matthews_corrcoef

# configuration des chemins relatifs
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from evaluation import (
    compute_challenge_metric,
    compute_f_measure,
)
from model_factory import get_shared_parser
from core_utils import init_eval_env, run_evaluation_inference, log_comprehensive_metrics


# variables globales pour le multiprocessing
_worker_probs = None
_worker_labels = None
_worker_weights = None
_worker_classes = None
_worker_sr = None


def _init_worker(probs, labels, weights, classes, sr):
    """Initialise la mémoire partagée pour chaque processus travailleur."""
    global _worker_probs, _worker_labels, _worker_weights, _worker_classes, _worker_sr
    _worker_probs = probs
    _worker_labels = labels
    _worker_weights = weights
    _worker_classes = classes
    _worker_sr = sr


def _val_batch(args):
    """
    Évalue les performances d'une configuration de seuils spécifique sur un sous-ensemble de données.
    Fonction conçue pour être exécutée en parallèle par les processus travailleurs lors de la recherche par quadrillage.

    Args:
        args (tuple): Un tuple contenant :
            - th (float): Le seuil testé pour la classe cible.
            - class_idx (int): L'index de la classe en cours d'optimisation.
            - base_thresholds (numpy.ndarray): Le vecteur des seuils de base pour les autres classes.

    Returns:
        tuple: Un tuple contenant le score du challenge, le F1-score macro, le score MCC, et la valeur du seuil testé (th).
    """
    th, class_idx, base_thresholds = args
    test_thresholds = base_thresholds.copy()
    test_thresholds[class_idx] = th

    binary_preds = (_worker_probs >= test_thresholds).astype(bool)

    challenge = compute_challenge_metric(_worker_weights, _worker_labels, binary_preds, _worker_classes, _worker_sr)
    f1, _ = compute_f_measure(_worker_labels, binary_preds)
    mcc = matthews_corrcoef(_worker_labels.flatten(), binary_preds.flatten())

    return challenge, f1, mcc, th


def optimize_all_metrics(val_labels, val_probs, weights, classes, nsr_index, num_workers, epochs=3):
    """
    Effectue une recherche multivariée itérative pour déterminer les vecteurs de seuils optimisant de manière indépendante plusieurs métriques cibles.
    
    Utilise le multiprocessing pour accélérer l'exploration de l'espace des paramètres. À chaque époque, la fonction itère sur chaque classe pour affiner le seuil de décision tout en maintenant les autres fixes.

    Args:
        val_labels (numpy.ndarray): Matrice des étiquettes réelles de l'ensemble de validation.
        val_probs (numpy.ndarray): Matrice des probabilités issues de l'inférence du modèle.
        weights (numpy.ndarray): Matrice des poids pour le calcul de la métrique du challenge.
        classes (list): Liste des noms des pathologies.
        nsr_index (int): Index associé au rythme sinusal normal.
        num_workers (int): Nombre de processus parallèles à allouer pour l'évaluation.
        epochs (int): Nombre de passages complets d'optimisation sur l'ensemble des classes.

    Returns:
        dict: Un dictionnaire associant chaque métrique cible (score du challenge, F1-score macro, MCC) à son meilleur score obtenu et à son vecteur de seuils optimal.
    """
    nsr_name = classes[nsr_index]
    best_thresholds = np.ones(len(classes)) * 0.5
    candidate_thresholds = np.arange(0.05, 0.51, 0.01)

    best_states = {
        "challenge_score": {"score": -np.inf, "thresholds": best_thresholds.copy()},
        "macro_f1": {"score": -np.inf, "thresholds": best_thresholds.copy()},
        "mcc": {"score": -np.inf, "thresholds": best_thresholds.copy()}
    }

    pool_args = (val_probs, val_labels, weights, classes, nsr_name)

    print("\n[optimisation] lancement de la recherche multivariée...")
    with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=pool_args) as pool:
        for epoch in range(epochs):
            for i in tqdm(range(len(classes)), desc=f"optimisation - époque {epoch+1}/{epochs}"):
                tasks = [(th, i, best_thresholds) for th in candidate_thresholds]

                best_chal_local = -np.inf
                best_th_local = best_thresholds[i]

                for chal, f1, mcc, th in pool.imap_unordered(_val_batch, tasks):
                    current_th_vector = best_thresholds.copy()
                    current_th_vector[i] = th
    
                    if chal > best_chal_local:
                        best_chal_local = chal
                        best_th_local = th

                    if chal > best_states["challenge_score"]["score"]:
                        best_states["challenge_score"].update({"score": chal, "thresholds": current_th_vector.copy()})
                    if f1 > best_states["macro_f1"]["score"]:
                        best_states["macro_f1"].update({"score": f1, "thresholds": current_th_vector.copy()})
                    if mcc > best_states["mcc"]["score"]:
                        best_states["mcc"].update({"score": mcc, "thresholds": current_th_vector.copy()})

                best_thresholds[i] = best_th_local

    return best_states


def run(args):
    """
    Orchestre le processus global de post-validation et d'optimisation des seuils de décision.

    Configure l'environnement, charge le modèle depuis un point de sauvegarde (checkpoint), exécute l'inférence sur le jeu de validation, et lance la recherche multivariée. 
    Évalue ensuite la robustesse des résultats via bootstrapping, extrait les performances de la sous-tâche binaire, génère des graphiques d'analyse, exporte les configurations optimales au format JSON et synchronise l'ensemble des artefacts avec Weights & Biases.

    Args:
        args (argparse.Namespace): Arguments contenant la configuration système, les hyperparamètres et les chemins d'accès aux fichiers.
    """
    classes, weights, nsr_index = init_eval_env(args)
    labels, probs = run_evaluation_inference(args, args.val_data, "post_val", "validation")

    best_configs = optimize_all_metrics(labels, probs, weights, classes, nsr_index, epochs=3, num_workers=args.workers)
    
    thresholds_dict = {
        'challenge': best_configs["challenge_score"]["thresholds"],
        'f1': best_configs["macro_f1"]["thresholds"],
        'mcc': best_configs["mcc"]["thresholds"]
    }

    log_comprehensive_metrics(labels, probs, thresholds_dict, weights, classes, nsr_index, prefix="val")

    final_config = {"configurations_optimales": {}}
    for metric, data in best_configs.items():
        wandb.run.summary[f"optimal_{metric}"] = data["score"]
        final_config["configurations_optimales"][metric] = {
            "score": float(data["score"]),
            "seuils": {cls: float(t) for cls, t in zip(classes, data["thresholds"])}
        }

    config_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_config_opti.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(final_config, f, indent=4)

    # Création de l'artefact
    config_artifact = wandb.Artifact(name="configuration_seuils", type="config")
    # Ajout du fichier à l'artefact
    config_artifact.add_file(config_path)
    # Envoi de l'artefact vers Weights & Biases
    wandb.log_artifact(config_artifact)

    wandb.finish()



def main():
    shared_parser = get_shared_parser()
    parser = argparse.ArgumentParser(description="optimisation des seuils multivariée", parents=[shared_parser])
    parser.add_argument('--val_data', default="../../../output/final_data/val")
    parser.add_argument('-c', '--checkpoint', default=None)
    parser.add_argument('--weights', default="../../ressources/weights_abbreviations.csv", help="PhysioNet weights.csv")
    args = parser.parse_args()


    if args.checkpoint is None:
        search_pattern = os.path.join(args.checkpoint_dir, f"best_model_{args.model_name}*.pt")
        found_configs = glob.glob(search_pattern)

        if not found_configs:
            raise FileNotFoundError(f"Aucun fichier correspondant à {search_pattern} n'a été trouvé. Veuillez vérifier le chemin ou spécifier --checkpoint.")

        args.checkpoint = os.path.basename(found_configs[0])

        if len(found_configs) > 1:
            print(f"[WARNING] Plusieurs fichiers de checkpoint trouvés. Utilisation de : {args.checkpoint}")


    args.workers = max(4, mp.cpu_count()-1)
    run(args)

if __name__ == "__main__":
    main()
