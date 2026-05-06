import os
import time
import shutil
import glob
import json
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp

import wandb

from model_factory import get_shared_parser, build_model, create_dataloader
from core_utils import (
    setup_global_environment, 
    setup_wandb, 
    load_pos_weight, 
    load_checkpoint,
    run_inference,
    init_eval_env,
    log_comprehensive_metrics,
    NEED_COMPILE
)

from train import run_training_loop, generate_exp_name 
from post_val import optimize_all_metrics



def get_kfold_splits(files, k=9, seed=42):
    """
    Génère les indices pour une séparation stricte des données en ensembles
    d'entraînement, de validation et de test.

    Garantit trois sous-ensembles mutuellement exclusifs pour chaque pli.

    Args:
        files (np.ndarray): Tableau contenant les chemins des fichiers.
        k (int): Nombre de plis pour la validation croisée.
        seed (int): Graine de génération aléatoire pour la reproductibilité.

    Returns:
        list: Liste de tuples contenant les indices (train, val, test) pour chaque pli.
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(files))
    folds = np.array_split(indices, k)

    splits = []
    for i in range(k):
        test_idx = folds[i]
        val_idx = folds[(i + 1) % k]

        train_folds = [folds[j] for j in range(k) if j != i and j != (i + 1) % k]
        train_idx = np.concatenate(train_folds)

        splits.append((train_idx, val_idx, test_idx))

    return splits


def create_symlink_fold(fold_dir, train_files, val_files, test_files):
    """
    Crée les liens symboliques pointant vers les fichiers de données pour 
    les trois ensembles d'un pli spécifique.

    Args:
        fold_dir (str): Chemin du répertoire racine pour le pli.
        train_files (np.ndarray): Fichiers destinés à l'entraînement.
        val_files (np.ndarray): Fichiers destinés à la validation.
        test_files (np.ndarray): Fichiers destinés au test.

    Returns:
        tuple: Chemins des répertoires créés (train, val, test).
    """
    dirs = {
        "train": os.path.join(fold_dir, "train"),
        "val": os.path.join(fold_dir, "val"),
        "test": os.path.join(fold_dir, "test")
    }

    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)

    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)

    for f in train_files: os.symlink(f, os.path.join(dirs["train"], os.path.basename(f)))
    for f in val_files: os.symlink(f, os.path.join(dirs["val"], os.path.basename(f)))
    for f in test_files: os.symlink(f, os.path.join(dirs["test"], os.path.basename(f)))

    return dirs["train"], dirs["val"], dirs["test"]


def get_best_checkpoint(ckpt_dir):
    """
    Récupère le chemin du meilleur point de sauvegarde du modèle pour un pli donné.

    Args:
        ckpt_dir (str): Répertoire contenant les points de sauvegarde.

    Returns:
        str or None: Le chemin vers le fichier du meilleur modèle, ou None si inexistant.
    """
    ckpts = glob.glob(os.path.join(ckpt_dir, "best_model_*.pt"))
    return ckpts[0] if ckpts else None


def execute_post_val(args, model, val_loader, device, use_amp, amp_dtype, classes, weights, nsr_index, fold_ckpt_dir):
    """
    Exécute la phase de post-validation pour optimiser les seuils de classification 
    multivariée sur l'ensemble de validation.

    Args:
        args (argparse.Namespace): Arguments de configuration globale.
        model (torch.nn.Module): Le modèle pré-entraîné.
        val_loader (torch.utils.data.DataLoader): Chargeur de données de validation.
        device (torch.device): Périphérique d'exécution.
        use_amp (bool): Activation de la précision mixte.
        amp_dtype (torch.dtype): Type de données pour la précision mixte.
        classes (list): Liste des classes cibles.
        weights (np.ndarray): Poids associés aux classes pour la métrique de challenge.
        nsr_index (int): Index de la classe représentant le rythme sinusal normal.
        fold_ckpt_dir (str): Répertoire de sauvegarde spécifique au pli en cours.

    Returns:
        dict: Configuration contenant les seuils optimisés.
    """
    print("\n" + "-"*40 + "\n[Phase 2] Post-validation et optimisation\n" + "-"*40)
    labels, probs, is_valid = run_inference(model, val_loader, device, use_amp, amp_dtype, desc="[Post-validation]")

    if not is_valid:
        print("[Avertissement] Valeurs invalides détectées. Application de l'imputation (0.0).")
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

    best_configs = optimize_all_metrics(labels, probs, weights, classes, nsr_index, epochs=3, num_workers=args.workers)

    thresholds_dict = {
        'challenge': best_configs["challenge_score"]["thresholds"],
        'f1': best_configs["macro_f1"]["thresholds"],
        'mcc': best_configs["mcc"]["thresholds"]
    }
    log_comprehensive_metrics(labels, probs, thresholds_dict, weights, classes, nsr_index, prefix="val")

    final_config = {"configurations_optimales": {}}
    for metric, data in best_configs.items():
        wandb.run.summary[f"optimal_val_{metric}"] = data["score"]
        final_config["configurations_optimales"][metric] = {
            "score": float(data["score"]),
            "seuils": {cls: float(t) for cls, t in zip(classes, data["thresholds"])}
        }

    config_path = os.path.join(fold_ckpt_dir, f"{args.model_name}_config_opti.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(final_config, f, indent=4)

    return final_config


def execute_eval(args, model, test_loader, device, use_amp, amp_dtype, classes, weights, nsr_index, config_opti):
    """
    Exécute la phase d'évaluation finale sur l'ensemble de test en appliquant 
    les seuils optimisés.

    Args:
        args (argparse.Namespace): Arguments de configuration globale.
        model (torch.nn.Module): Le modèle pré-entraîné.
        test_loader (torch.utils.data.DataLoader): Chargeur de données de test.
        device (torch.device): Périphérique d'exécution.
        use_amp (bool): Activation de la précision mixte.
        amp_dtype (torch.dtype): Type de données pour la précision mixte.
        classes (list): Liste des classes cibles.
        weights (np.ndarray): Poids associés aux classes pour la métrique de challenge.
        nsr_index (int): Index de la classe représentant le rythme sinusal normal.
        config_opti (dict): Configuration contenant les seuils préalablement optimisés.

    Returns:
        dict: Métriques de performance évaluées sur l'ensemble de test.
    """
    print("\n" + "-"*40 + "\n[Phase 3] Évaluation finale (test)\n" + "-"*40)
    labels, probs, is_valid = run_inference(model, test_loader, device, use_amp, amp_dtype, desc="[Évaluation test]")

    if not is_valid:
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

    val_configs = config_opti["configurations_optimales"]
    thresholds_dict = {
        'challenge': np.array([val_configs["challenge_score"]["seuils"][c] for c in classes]),
        'f1': np.array([val_configs["macro_f1"]["seuils"][c] for c in classes]),
        'mcc': np.array([val_configs["mcc"]["seuils"][c] for c in classes])
    }

    return log_comprehensive_metrics(labels, probs, thresholds_dict, weights, classes, nsr_index, prefix="test")


def run_kfold_pipeline(args):
    """
    Orchestre l'ensemble du pipeline de validation croisée, incluant la gestion 
    de la persistance, la reprise sur erreur et l'agrégation statistique des métriques.

    Args:
        args (argparse.Namespace): Arguments de configuration issus de la ligne de commande.
    """
    print(f"[Initialisation] Recherche dans : {args.data_dirs}")
    all_files = np.array(sorted([str(p.resolve()) for p in Path(args.data_dirs).rglob('*signal.npy') if p.is_file()]))

    if len(all_files) == 0: 
        raise ValueError(f"Aucun fichier trouvé dans {args.data_dirs}")

    device, use_amp, amp_dtype = setup_global_environment(args)
    classes, weights, nsr_index = init_eval_env(args) 

    master_id_file = os.path.join(args.checkpoint_dir, f"kfold_master_id_{args.model_name}.txt")
    if os.path.exists(master_id_file):
        with open(master_id_file, 'r') as f: group_id = f.read().strip()
    else:
        group_id = wandb.util.generate_id()[:6]
        with open(master_id_file, 'w') as f: f.write(group_id)

    group_name = f"kfold_{args.model_name}_{group_id}"
    kfold_checkpoint_root = os.path.join(args.checkpoint_dir, group_name)
    os.makedirs(kfold_checkpoint_root, exist_ok=True)
    base_tmp_data = os.path.join(args.output, f"tmp_kfold_{group_id}")

    kf_splits = get_kfold_splits(all_files, k=args.k)
    total_exp_start = time.time()

    for fold, (train_idx, val_idx, test_idx) in enumerate(kf_splits):
        print(f"\n{'='*60}\nDémarrage du pli {fold}/{args.k - 1}\n{'='*60}")
        fold_ckpt_dir = os.path.join(kfold_checkpoint_root, f"fold_{fold}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)

        done_flag = os.path.join(fold_ckpt_dir, "fold_completed.json")
        if os.path.exists(done_flag):
            print(f"[Information] Pli {fold} déjà complété. Passage au suivant.")
            continue

        wandb_id_file = os.path.join(fold_ckpt_dir, "wandb_id.txt")
        if os.path.exists(wandb_id_file):
            with open(wandb_id_file, 'r') as f: fold_wandb_id = f.read().strip()
        else:
            fold_wandb_id = wandb.util.generate_id()
            with open(wandb_id_file, 'w') as f: f.write(fold_wandb_id)

        setup_wandb(args, job_type=f"fold_{fold}", run_name=f"f{fold}_{group_id}", group=group_name, wandb_id=fold_wandb_id)

        fold_data_dir = os.path.join(base_tmp_data, f"fold_{fold}")
        train_path, val_path, test_path = create_symlink_fold(fold_data_dir, all_files[train_idx], all_files[val_idx], all_files[test_idx])

        model, valid_kwargs, Dataset_fun, gen_fun = build_model(args)
        model = model.to(device)

        if args.model_name in NEED_COMPILE or args.use_static_padding:
            try: 
                model = torch.compile(model)
            except Exception as e: 
                print(f"[Avertissement] Échec de la compilation du modèle : {e}")

        param_groups = [
            {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": args.backbone_lr},
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": args.lr}
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay, fused=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=load_pos_weight(args.pos_weight_path, args.num_classes, device))
        scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16)) if use_amp else None

        train_loader = create_dataloader(args, train_path, Dataset_fun, gen_fun, is_train=True)
        val_loader = create_dataloader(args, val_path, Dataset_fun, gen_fun, is_train=False)
        test_loader = create_dataloader(args, test_path, Dataset_fun, gen_fun, is_train=False)

        start_epoch, best_score = 1, -1.0
        if args.resume_from:
            resume_path = os.path.join(args.checkpoint_dir, args.resume_from)
            if os.path.exists(resume_path):
                print(f"[Reprise] Restauration depuis le fichier spécifié : {resume_path}")
                start_epoch, best_score = load_checkpoint(resume_path, model, device, optimizer, scaler)
                args.resume_from = None 
            else:
                print(f"[Erreur] Le fichier spécifié pour la reprise n'existe pas : {resume_path}")
                print("[Information] Démarrage de l'entraînement initial.")
                args.resume_from = None

        original_ckpt_dir = args.checkpoint_dir
        args.checkpoint_dir = fold_ckpt_dir

        if start_epoch <= args.epochs:
            exp_name = generate_exp_name(args, valid_kwargs, fold_wandb_id)
            run_training_loop(
                args=args, model=model, train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion, scaler=scaler, device=device,
                use_amp=use_amp, amp_dtype=amp_dtype, exp_name=exp_name,
                start_epoch=start_epoch, best_val_pr_auc=best_score, fold=fold
            )

        best_ckpt = get_best_checkpoint(fold_ckpt_dir)
        if best_ckpt:
            load_checkpoint(best_ckpt, model, device)

        config_opti = execute_post_val(args, model, val_loader, device, use_amp, amp_dtype, classes, weights, nsr_index, fold_ckpt_dir)
        test_metrics = execute_eval(args, model, test_loader, device, use_amp, amp_dtype, classes, weights, nsr_index, config_opti)

        args.checkpoint_dir = original_ckpt_dir 

        with open(done_flag, 'w') as f: 
            json.dump({
                "status": "completed", 
                "time": time.time(),
                "test_metrics": test_metrics
            }, f, indent=4)

        wandb.finish()
        shutil.rmtree(fold_data_dir)

    print("\n" + "="*60 + "\n[Agrégation et synchronisation WandB]\n" + "="*60)
    
    all_metrics = {"challenge_score": [], "macro_f1": [], "mcc": []}
    completed_folds = 0

    for fold in range(args.k):
        done_flag = os.path.join(kfold_checkpoint_root, f"fold_{fold}", "fold_completed.json")
        if os.path.exists(done_flag):
            with open(done_flag, 'r') as f:
                data = json.load(f)
                if "test_metrics" in data:
                    completed_folds += 1
                    for k_metric, v_metric in data["test_metrics"].items():
                        if k_metric in all_metrics:
                            all_metrics[k_metric].append(v_metric)

    if completed_folds > 0:
        setup_wandb(args, job_type="kfold_summary", run_name=f"SYNTHESE_kfold_{group_id}", group=group_name)
        
        summary_payload = {}
        for metric_name, values in all_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            var_val = np.var(values)
            
            summary_payload[f"global_kfold/mean_{metric_name}"] = mean_val
            summary_payload[f"global_kfold/std_{metric_name}"] = std_val
            summary_payload[f"global_kfold/var_{metric_name}"] = var_val

            print(f"- {metric_name} : Moyenne = {mean_val:.4f} ± {std_val:.4f}")
            
        wandb.log(summary_payload)
        wandb.finish()
        print("[Information] Synthèse envoyée avec succès sur WandB.")
    else:
        print("[Avertissement] Aucun résultat de test trouvé pour générer le bilan.")

    total_time = (time.time() - total_exp_start) / 3600
    print(f"\n[Terminé] Temps total d'exécution : {total_time:.2f} heures.")

    if os.path.exists(base_tmp_data):
        shutil.rmtree(base_tmp_data)


def main():
    """
    Point d'entrée principal du script. 
    Initialise l'analyseur d'arguments et déclenche le pipeline de validation croisée.
    """
    shared_parser = get_shared_parser()
    parser = argparse.ArgumentParser(description="Pipeline de validation croisée (Entraînement -> Optimisation -> Évaluation)", parents=[shared_parser])

    parser.add_argument('-k', type=int, default=9, help="Nombre de plis pour la validation croisée.")
    parser.add_argument('--data_dirs', type=str, default="../../../output/final_data/")
    parser.add_argument('--pos_weight_path', type=str, default="../ressources/pos_weight.pt")

    parser.add_argument('--resume_from', type=str, default=None, 
                        help="Nom du fichier .pt pour reprendre l'entraînement.")

    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--val_start_epoch', type=int, default=10)

    args = parser.parse_args()
    args.workers = max(4, mp.cpu_count() - 1)

    run_kfold_pipeline(args)


if __name__ == "__main__":
    main()
