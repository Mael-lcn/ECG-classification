import os, glob
import sys
import json
from tqdm import tqdm
import argparse
import numpy as np

from sklearn.metrics import matthews_corrcoef, hamming_loss
from sklearn.utils import resample

import wandb

project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from model_factory import get_shared_parser
from core_utils import init_eval_env, run_evaluation_inference, log_comprehensive_metrics



def validate_bootstrapping(labels, probs, weights, classes, nsr_index, thresholds, n_iterations=100):
    """
    Évalue la robustesse statistique des métriques d'évaluation via la méthode de bootstrapping. 
    
    Calcule les intervalles de confiance à 95 % pour le score du challenge, le F1-score macro et la perte de Hamming,
    en effectuant des tirages aléatoires avec remise sur les données de validation.

    Args:
        labels (numpy.ndarray): Matrice des étiquettes réelles.
        probs (numpy.ndarray): Matrice des probabilités prédites.
        weights (numpy.ndarray): Matrice des pondérations pour le score du challenge.
        classes (list): Liste des désignations des classes.
        nsr_index (int): Index de la classe correspondant au rythme sinusal normal.
        thresholds (numpy.ndarray): Vecteur des seuils de décision optimisés par classe.
        n_iterations (int): Nombre d'itérations pour le tirage de bootstrapping.

    Returns:
        tuple: Un tuple contenant :
            - results (dict): Les métriques statistiques agrégées (moyenne et intervalles de confiance pour chaque métrique).
            - stats (dict): Les listes des scores bruts calculés pour chaque itération de la boucle.
    """
    n_samples = len(labels)
    nsr_name = classes[nsr_index]

    stats = {
        'challenge': [],
        'macro_f1': [],
        'hamming_loss': []
    }

    print(f"\n[analyse] Lancement du bootstrapping statistique ({n_iterations} itérations)...")
    for _ in tqdm(range(n_iterations), desc="Bootstrapping", leave=False):
        # Tirage avec remise
        indices = resample(np.arange(n_samples), replace=True)
        sample_labels = labels[indices]
        sample_probs = probs[indices]

        sample_preds = (sample_probs >= thresholds).astype(bool)

        stats['challenge'].append(compute_challenge_metric(weights, sample_labels, sample_preds, classes, nsr_name))
        macro_f1, _ = compute_f_measure(sample_labels, sample_preds)
        stats['macro_f1'].append(macro_f1)
        stats['hamming_loss'].append(hamming_loss(sample_labels, sample_preds))

    results = {}
    for metric, values in stats.items():
        results[f"stabilité/{metric}_mean"] = np.mean(values)
        results[f"stabilité/{metric}_ci_lower"] = np.percentile(values, 2.5)
        results[f"stabilité/{metric}_ci_upper"] = np.percentile(values, 97.5)

    return results, stats


def compute_binary_metrics(labels, probs, thresholds, nsr_index):
    """
    Calcule les métriques de classification binaire pour la sous-tâche distinguant les patients sains des patients malades.
    
    Un patient est considéré comme malade s'il présente au moins une pathologie autre que le rythme sinusal normal (NSR).

    Args:
        labels (numpy.ndarray): Matrice binaire des étiquettes réelles.
        probs (numpy.ndarray): Matrice des probabilités prédites.
        thresholds (numpy.ndarray): Tableau des seuils de décision par classe.
        nsr_index (int): Index de la classe représentant l'état sain.

    Returns:
        tuple: Un tuple contenant :
            - metrics (dict): Dictionnaire comprenant la sensibilité, spécificité, F1-score, coefficient de corrélation de Matthews (MCC) et la prévalence de la maladie.
            - real_sick (numpy.ndarray): Vecteur booléen indiquant la présence réelle d'une pathologie.
            - pred_sick (numpy.ndarray): Vecteur booléen indiquant la prédiction d'une pathologie par le modèle.
    """
    num_classes = labels.shape[1]
    other_indices = [i for i in range(num_classes) if i != nsr_index]

    real_sick = np.any(labels[:, other_indices] == 1, axis=1)
    preds = (probs >= thresholds).astype(bool)
    pred_sick = np.any(preds[:, other_indices], axis=1)

    tp = np.sum((real_sick == 1) & (pred_sick == 1))
    tn = np.sum((real_sick == 0) & (pred_sick == 0))
    fp = np.sum((real_sick == 0) & (pred_sick == 1))
    fn = np.sum((real_sick == 1) & (pred_sick == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_binary = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    mcc = matthews_corrcoef(real_sick, pred_sick)

    return {
        "binaire_malade/sensibilité": sensitivity,
        "binaire_malade/spécificité": specificity,
        "binaire_malade/f1_score": f1_binary,
        "binaire_malade/mcc": mcc,
        "binaire_malade/prevalence": np.mean(real_sick)
    }, real_sick, pred_sick


def compute_accuracy(labels, outputs):
    num_recordings = labels.shape[0]
    correct = np.all(labels == outputs, axis=1)
    return float(np.sum(correct)) / float(num_recordings)

def compute_confusion_matrices(labels, outputs, normalize=False):
    num_recordings, num_classes = np.shape(labels)
    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: A[j, 0, 0] += 1
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: A[j, 0, 0] += 1.0/normalization
    return A

def compute_f_measure(labels, outputs):
    num_recordings, num_classes = np.shape(labels)
    A = compute_confusion_matrices(labels, outputs)
    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn: f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else: f_measure[k] = float('nan')

    macro_f_measure = np.nanmean(f_measure) if np.any(np.isfinite(f_measure)) else float('nan')
    return macro_f_measure, f_measure

def compute_auc(labels, outputs):
    num_recordings, num_classes = labels.shape
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]

        tp, fp, fn, tn = np.zeros(len(thresholds)), np.zeros(len(thresholds)), np.zeros(len(thresholds)), np.zeros(len(thresholds))
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        idx = np.argsort(outputs[:, k])[::-1]
        i = 0

        for j in range(1, len(thresholds)):
            tp[j], fp[j], fn[j], tn[j] = tp[j-1], fp[j-1], fn[j-1], tn[j-1]
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        tpr = tp / np.maximum(tp + fn, 1)
        tnr = tn / np.maximum(fp + tn, 1)
        ppv = tp / np.maximum(tp + fp, 1)

        for j in range(len(thresholds) - 1):
            auroc[k] += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
            auprc[k] += (tpr[j+1] - tpr[j]) * ppv[j+1]

    return np.nanmean(auroc), np.nanmean(auprc), auroc, auprc

def compute_modified_confusion_matrix(labels, outputs):
    num_recordings, num_classes = labels.shape
    A = np.zeros((num_classes, num_classes))
    for i in range(num_recordings):
        union_count = np.sum(labels[i] | outputs[i])
        if union_count > 0:
            A += np.outer(labels[i], outputs[i]) / float(union_count)
    return A

def compute_challenge_metric(weights, labels, outputs, classes, sinus_rhythm):
    num_recordings, num_classes = labels.shape
    if sinus_rhythm in classes:
        sinus_rhythm_index = classes.index(sinus_rhythm)
    else:
        raise ValueError('The sinus rhythm class is not available.')

    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool_)
    inactive_outputs[:, sinus_rhythm_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score

def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

def is_finite_number(x):
    return np.isfinite(float(x)) if is_number(x) else False

def load_table(table_file):
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    num_rows = len(table)-1
    if num_rows<1: raise Exception('The table {} is empty.'.format(table_file))
    row_lengths = set(len(table[i])-1 for i in range(num_rows))
    if len(row_lengths)!=1: raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(row_lengths)
    if num_cols<1: raise Exception('The table {} is empty.'.format(table_file))

    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            values[i, j] = float(value) if is_finite_number(value) else float('nan')

    return rows, cols, values

def load_weights(weights_file):
    rows, cols, values = load_table(weights_file)
    assert(rows == cols)
    return rows, values

def round_val(val, digits=5):
    try:
        if val is not None and np.isfinite(val): return round(float(val), digits)
        return val
    except:
        return val


def run(args):
    """
    Orchestre le processus global d'évaluation sur le jeu de test.

    Configure l'environnement, charge le modèle depuis un point de sauvegarde (checkpoint)
    et exécute l'inférence sur le jeu de données de test. 
    Applique ensuite les seuils optimaux (précédemment extraits du fichier de configuration)
    pour évaluer les performances globales via de multiples métriques (score du challenge, F1-score macro, MCC, AUC, AUPRC). 
    Analyse également la robustesse statistique via bootstrapping, extrait les performances
    de la sous-tâche binaire (sain contre malade), évalue les performances détaillées par pathologie,
    et synchronise l'ensemble des résultats et graphiques d'analyse avec wandb.

    Args:
        args (argparse.Namespace): Arguments contenant la configuration du système,
        les hyperparamètres et les chemins d'accès aux fichiers (données de test, poids, fichier de configuration des seuils).
    """
    classes, weights, nsr_index = init_eval_env(args)
    labels, probs = run_evaluation_inference(args, args.test_data, "test_eval", "test")

    with open(args.config_file) as f:
        data = json.load(f)
        val_configs = data.get("configurations_optimales", data)

    thresholds_dict = {
        'challenge': np.array([val_configs["challenge_score"]["seuils"][c] for c in classes]),
        'f1': np.array([val_configs["macro_f1"]["seuils"][c] for c in classes]),
        'mcc': np.array([val_configs["mcc"]["seuils"][c] for c in classes])
    }

    log_comprehensive_metrics(labels, probs, thresholds_dict, weights, classes, nsr_index, prefix="test")

    wandb.finish()



def main():
    shared_parser = get_shared_parser()
    parser = argparse.ArgumentParser(description="Script d'évaluation finale", parents=[shared_parser])
    parser.add_argument('--test_data', default="../../../../output/final_data/test", help="HDF5 test dataset directory")
    parser.add_argument('-c', '--checkpoint', default=None, help="name of the checkpoint.pt file")
    parser.add_argument('--weights', default="../../ressources/weights_abbreviations.csv", help="PhysioNet weights.csv")
    parser.add_argument('--config_file', type=str, default=None, help="Name of the config file")

    args = parser.parse_args()

    if args.config_file is None:
        search_pattern = os.path.join(args.checkpoint_dir, "*config_opti.json")
        found_configs = glob.glob(search_pattern)

        if not found_configs:
            raise FileNotFoundError(f"Aucun fichier correspondant à {search_pattern} n'a été trouvé. Veuillez vérifier le chemin ou spécifier --config_file.")

        args.config_file = found_configs[0]

        if len(found_configs) > 1:
            print(f"[WARNING] Plusieurs fichiers de configuration trouvés. Utilisation de : {args.config_file}")
    else:
        args.config_file = os.path.join(args.checkpoint_dir, args.config_file)


    if args.checkpoint is None:
        search_pattern = os.path.join(args.checkpoint_dir, f"best_model_{args.model_name}*.pt")
        found_configs = glob.glob(search_pattern)

        if not found_configs:
            raise FileNotFoundError(f"Aucun fichier correspondant à {search_pattern} n'a été trouvé. Veuillez vérifier le chemin ou spécifier --checkpoint.")

        args.checkpoint = os.path.basename(found_configs[0])

        if len(found_configs) > 1:
            print(f"[WARNING] Plusieurs fichiers de checkpoint trouvés. Utilisation de : {args.checkpoint}")

    run(args)


if __name__ == "__main__":
    main()
