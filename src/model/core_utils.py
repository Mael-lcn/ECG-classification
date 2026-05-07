import os, time
import numpy as np
import torch
import wandb
from tqdm import tqdm
import json
import multiprocessing as mp
from sklearn.metrics import matthews_corrcoef


NEED_COMPILE = set(['PatchTSTModel', 'DinoTraceTemporal', 'ViT_TimeFreq', 'ViT_Image', 'CNN_Image'])


def setup_global_environment(args):
    """
    Initialise l'environnement matériel, la gestion de la mémoire et la précision mixte (AMP).

    Configure l'allocation mémoire de PyTorch pour limiter la fragmentation, 
    crée les dossiers de sauvegarde nécessaires, et configure le périphérique cible (GPU/CPU). 
    Active et optimise l'Automatic Mixed Precision (AMP) si le matériel le supporte (ex: BFloat16 sur Ampere).

    Args:
        args (argparse.Namespace): Objets d'arguments contenant les configurations globales 
            (ex: `checkpoint_dir`, `output`, `gpu`, `not_use_amp`, `use_static_padding`).

    Returns:
        tuple: Un tuple contenant :
            - device (torch.device): Le périphérique de calcul alloué (ex: 'cuda:0' ou 'cpu').
            - use_amp (bool): Indique si la précision mixte est activée.
            - amp_dtype (torch.dtype): Le type de tenseur utilisé pour l'AMP (torch.float16 ou torch.bfloat16).
    """
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    torch.set_float32_matmul_precision('high')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        device = torch.device(f"cuda:{args.gpu}")
        use_amp = not getattr(args, 'not_use_amp', False)
        
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        if use_amp:
            print(f"[INIT] AMP activé ({'BFloat16' if amp_dtype == torch.bfloat16 else 'Float16'})")

        torch.backends.cudnn.benchmark = getattr(args, 'use_static_padding', False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    else:
        device, use_amp, amp_dtype = torch.device("cpu"), False, torch.float32
        print("[INIT] Mode CPU forcé.")

    return device, use_amp, amp_dtype


def setup_wandb(args, job_type, run_name, group=None, wandb_id=None, resume_mode="allow", tags=None):
    """
    Initialise une session Weights & Biases (WandB) de manière modulaire et isolée.

    Configure un environnement WandB en mode hors ligne ('offline') pour le suivi des 
    expériences, gère la création des répertoires de cache temporaires et initialise 
    le run avec les tags et les configurations appropriés.

    Args:
        args (argparse.Namespace): Paramètres globaux de l'expérience (fournira la configuration à WandB).
        job_type (str): Le type de tâche exécutée (ex: 'train', 'eval', 'post_val').
        run_name (str): Le nom d'affichage de l'expérience sur le tableau de bord WandB.
        group (str, optional): Le nom du groupe pour regrouper plusieurs runs (par défaut, utilise `args.model_name`).
        wandb_id (str, optional): Identifiant unique du run pour la reprise. Généré automatiquement si None.
        resume_mode (str, optional): Stratégie de reprise du run ('allow', 'must', 'never').
        tags (list of str, optional): Liste de tags supplémentaires pour filtrer les expériences.

    Returns:
        str: L'identifiant unique (ID) du run WandB initialisé.
    """
    if wandb_id is None:
        wandb_id = wandb.util.generate_id()

    base_wandb_path = os.path.join(args.output, "wandb_logs")
    os.makedirs(base_wandb_path, exist_ok=True)

    os.environ["WANDB_MODE"] = "offline" 
    os.environ["WANDB_DIR"] = base_wandb_path
    os.environ["WANDB_CACHE_DIR"] = os.path.join(base_wandb_path, "cache")
    os.environ["TMPDIR"] = os.path.join(base_wandb_path, "tmp")
    os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    default_tags = ["offline", args.model_name]
    if tags: default_tags.extend(tags)

    wandb.init(
        project="ECG_Classification_Experiments",
        group=group or args.model_name,
        job_type=job_type,  # Ex: "train", "eval", "post_val"
        name=run_name,
        config=vars(args),
        id=wandb_id,
        resume=resume_mode,
        tags=list(set(default_tags))
    )
    return wandb_id


def create_attention_mask(valid_lens, max_len, device):
    """
    Fonction universelle pour créer un masque d'attention 2D [batch_size, max_len].
    Fonctionne pour les ECG 1D (valid_lens = batch_lens, max_len = target_t)
    ET pour les Images 2D (valid_lens = num_windows, max_len = target_num_images).
    
    True = Valide (garder), False = Padding (ignorer)
    """
    steps = torch.arange(max_len, device=device).unsqueeze(0)
    valid_lens = valid_lens.to(device, non_blocking=True).unsqueeze(1)

    return (steps < valid_lens).to(torch.float32)


@torch.no_grad()
def run_inference(model, dataloader, device, use_amp, amp_dtype, desc="[Inférence]", squeeze_batch=False):
    """
    Exécute une boucle d'inférence universelle sans calcul de gradient.

    Cette fonction peut être utilisée de manière transversale pour la validation, l'évaluation 
    finale ou la post-validation. Elle gère la précision mixte (AMP), l'extraction des probabilités 
    via une fonction sigmoïde, et surveille la validité numérique des prédictions (NaN/Inf).

    Args:
        model (torch.nn.Module): Le modèle d'apprentissage profond à évaluer.
        dataloader (torch.utils.data.DataLoader): Le chargeur de données fournissant les batchs.
        device (torch.device): Le périphérique matériel utilisé pour l'inférence.
        use_amp (bool): Active l'Automatic Mixed Precision pour accélérer l'inférence.
        amp_dtype (torch.dtype): Type de données utilisé par l'AMP (float16 ou bfloat16).
        desc (str, optional): Description affichée sur la barre de progression tqdm.
        squeeze_batch (bool, optional): Si True, retire la dimension supplémentaire ajoutée par 
            certains dataloaders (ex: lorsque batch_size=1 renvoie [1, Batch, ...]).

    Returns:
        tuple: Un tuple contenant :
            - labels_out (numpy.ndarray): Tableau booléen des étiquettes cibles.
            - probs_out (numpy.ndarray): Tableau des probabilités prédites (valeurs entre 0 et 1).
            - is_valid (bool): Vaut False si des valeurs NaN ou Inf ont été détectées dans les logits.
    """
    model.eval()
    all_labels, all_probs = [], []
    is_valid = True

    for x, y, batch_lens in tqdm(dataloader, desc=desc, leave=False):
        if squeeze_batch and x.shape[0] == 1:
            x = x.squeeze(0)
            y = y.squeeze(0)
            batch_lens = batch_lens.squeeze(0)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_mask = create_attention_mask(batch_lens, x.shape[2], device)

        if x.shape[-1] == 0:
            continue # Skip les données invalides

        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp, dtype=amp_dtype):
            logits = model(x, batch_mask=batch_mask)

            if not torch.isfinite(logits).all():
                is_valid = False

        probs = torch.sigmoid(logits)

        all_labels.append(y.cpu())
        all_probs.append(probs.cpu())

    labels_out = torch.cat(all_labels).numpy().astype(bool)
    probs_out = torch.cat(all_probs).float().numpy()

    return labels_out, probs_out, is_valid



def load_pos_weight(pos_weight_path, num_classes, device):
    if not pos_weight_path or not os.path.exists(pos_weight_path):
        return None
    pw = torch.load(pos_weight_path, map_location=device)
    if pw.shape != (num_classes,): return None
    return pw.sqrt().to(device)


def load_checkpoint(checkpoint_path, model, device, optimizer=None, scheduler=None, scaler=None):
    """
    Restaure l'état d'un modèle et optionnellement de son environnement d'entraînement à partir d'un point de sauvegarde.
    Gère de manière robuste le chargement des poids du modèle (en nettoyant les éventuels 
    préfixes `_orig_mod.` issus de `torch.compile`), de l'optimiseur, et du scaler AMP.

    Args:
        checkpoint_path (str): Le chemin absolu ou relatif vers le fichier `.pt` du checkpoint.
        model (torch.nn.Module): Le modèle dans lequel charger les poids.
        device (str or torch.device, optional): Périphérique cible pour le chargement.
        optimizer (torch.optim.Optimizer, optional): L'optimiseur à restaurer.
        scaler (torch.amp.GradScaler, optional): Le scaler de précision mixte à restaurer.

    Returns:
        tuple: Un tuple contenant :
            - start_epoch (int): L'époque à laquelle reprendre l'entraînement (1 si aucun checkpoint).
            - best_score (float): Le meilleur score de validation enregistré dans le checkpoint (-1.0 si absent).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERREUR] Checkpoint introuvable : {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 1. Modèle : Nettoyage des clés
    raw_state_dict = checkpoint.get('model_state_dict', checkpoint)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw_state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_score = checkpoint.get('best_val_pr_auc', checkpoint.get('best_score', -1.0))

    # 2. Optimiseur : Synchro pour fused=True
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for p, state in optimizer.state.items():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if v.shape == p.shape:
                        # Clone l'empreinte exacte (Type, Device, Contiguïté)
                        new_v = torch.empty_like(p, memory_format=torch.preserve_format)
                        state[k] = new_v.copy_(v)
                    else:
                        # Tenseurs 0-D (comme 'step'), on force sur le GPU
                        state[k] = v.to(device=p.device)

    # 3. Scheduler & Scaler
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    print(f"[REPRISE] État synchronisé : Epoch {start_epoch}, Score: {best_score:.4f}")
    return start_epoch, best_score



def init_eval_env(args):
    """Initialise le multiprocessing et charge les métadonnées (classes, poids, nsr)."""
    from evaluation import load_weights

    mp.set_start_method("spawn", force=True)
    torch.set_num_threads(args.workers)

    with open(args.class_map) as f:
        classes = json.load(f)

    weight_classes, weights = load_weights(args.weights)
    assert weight_classes == classes, "Mismatch entre class_map et weights.csv"

    return classes, weights, classes.index("NSR")


def run_evaluation_inference(args, data_path, job_type, run_name_prefix):
    """Gère l'environnement, le modèle, l'inférence et le nettoyage des NaNs."""
    from model_factory import build_model, create_dataloader

    device, use_amp, amp_dtype = setup_global_environment(args)

    wandb_id = setup_wandb(
        args=args, job_type=job_type,
        run_name=f"{run_name_prefix}_{args.model_name}_{wandb.util.generate_id()[:6]}",
        tags=[job_type, args.model_name]
    )

    model, _, Dataset_fun, gen_fun = build_model(args)
    model = model.to(device)

    load_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint), model, device=device)
    loader = create_dataloader(args, data_path, Dataset_fun, gen_fun, is_train=False)

    start_time = time.time()
    labels, probs, is_valid = run_inference(model, loader, device, use_amp, amp_dtype, desc=f"[{job_type}]")
    inference_time = time.time() - start_time

    if not is_valid:
        print(f"[WARNING] NaNs détectés lors de l'inférence. Application de l'imputation punitive (0.0).")
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

    wandb.log({
        "performance/temps_inference_sec": inference_time,
        "performance/fps": len(labels) / inference_time if inference_time > 0 else 0
    })

    return labels, probs

def log_comprehensive_metrics(labels, probs, thresholds_dict, weights, classes, nsr_index, prefix="eval"):
    """
    Calcule et loggue absolument toutes les métriques sur WandB.
    thresholds_dict doit contenir: {'challenge': array, 'f1': array, 'mcc': array}
    """
    from evaluation import (
        compute_challenge_metric, compute_f_measure, compute_accuracy, 
        compute_auc, validate_bootstrapping, compute_binary_metrics
    )

    # 1. Prédictions selon les seuils
    preds_chal = (probs >= thresholds_dict['challenge']).astype(bool)
    preds_f1 = (probs >= thresholds_dict['f1']).astype(bool)
    preds_mcc = (probs >= thresholds_dict['mcc']).astype(bool)

    # 2. Métriques Globales
    mean_brier = np.mean((probs - labels)**2)
    mean_auroc, mean_auprc, auroc_per_class, auprc_per_class = compute_auc(labels, probs)
    chal_score = compute_challenge_metric(weights, labels, preds_chal, classes, classes[nsr_index])
    macro_f1, f1_per_class = compute_f_measure(labels, preds_f1)

    wandb.log({
        f"{prefix}_metrics/challenge_score": chal_score,
        f"{prefix}_metrics/macro_f1": macro_f1,
        f"{prefix}_metrics/mcc": matthews_corrcoef(labels.flatten(), preds_mcc.flatten()),
        f"{prefix}_metrics/acc_exact_match": compute_accuracy(labels, preds_chal),
        f"{prefix}_metrics/mean_auroc": mean_auroc,
        f"{prefix}_metrics/mean_auprc": mean_auprc,
        f"{prefix}_metrics/mean_brier_score": mean_brier 
    })

    # 3. Bootstrapping
    robustness_metrics, raw_bootstrap = validate_bootstrapping(labels, probs, weights, classes, nsr_index, thresholds_dict['challenge'])
    wandb.log(robustness_metrics)
    wandb.log({
        "distribution_bootstrap/challenge": wandb.Histogram(raw_bootstrap['challenge']),
        "distribution_bootstrap/macro_f1": wandb.Histogram(raw_bootstrap['macro_f1'])
    })

    # 4. Tâche Binaire (Sain vs Malade)
    binary_results, real_sick, pred_sick = compute_binary_metrics(labels, probs, thresholds_dict['challenge'], nsr_index)
    wandb.log(binary_results)
    wandb.log({"analyse/matrice_confusion_binaire": wandb.plot.confusion_matrix(
        preds=pred_sick.astype(int), y_true=real_sick.astype(int), class_names=["Sain (NSR)", "Malade"]
    )})

    # 5. Métriques par Classe
    brier_scores_per_class = np.mean((probs - labels)**2, axis=0)
    tp = np.sum((labels == 1) & (preds_f1 == 1), axis=0)
    tn = np.sum((labels == 0) & (preds_f1 == 0), axis=0)
    fp = np.sum((labels == 0) & (preds_f1 == 1), axis=0)
    fn = np.sum((labels == 1) & (preds_f1 == 0), axis=0)
    recalls = tp / (tp + fn + 1e-9)
    specificities = tn / (tn + fp + 1e-9)

    table_data = [
        [classes[i], f1_per_class[i], auroc_per_class[i], auprc_per_class[i], recalls[i], specificities[i], brier_scores_per_class[i]] 
        for i in range(len(classes))
    ]
    wandb.log({
        "analyse/performances_par_classe": wandb.Table(
            columns=["Pathologie", "F1 Score", "AUROC", "AUPRC", "Sensibilité", "Spécificité", "Brier Score"], 
            data=table_data
        ),
        "analyse/f1_par_classe_bar": wandb.plot.bar(
            wandb.Table(columns=["Pathologie", "F1 Score"], data=[[c, f] for c, f in zip(classes, f1_per_class)]), 
            "Pathologie", "F1 Score", title="F1 Score par Pathologie"
        )
    })

    return {
        "challenge_score": float(chal_score),
        "macro_f1": float(macro_f1),
        "mcc": float(matthews_corrcoef(labels.flatten(), preds_mcc.flatten()))
    }
