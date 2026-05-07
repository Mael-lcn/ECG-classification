import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import h5py
import torchvision.utils as vutils



# --- Palette de 12 couleurs très distinctes (Format RGB) ---
distinct_colors = [
    (255, 0, 0),      # 1. Rouge
    (0, 0, 255),      # 2. Bleu profond
    (0, 150, 0),      # 3. Vert foncé
    (255, 120, 0),    # 4. Orange
    (150, 0, 150),    # 5. Violet
    (0, 200, 200),    # 6. Cyan (Turquoise)
    (139, 69, 19),    # 7. Marron (SaddleBrown)
    (255, 0, 255),    # 8. Magenta
    (100, 200, 0),    # 9. Vert Lime
    (0, 100, 100),    # 10. Sarcelle (Teal)
    (255, 105, 180),  # 11. Rose (Hot Pink)
    (0, 0, 128)       # 12. Bleu Marine
]


def create_image_12leads_together(tracings, lengths=None, h=512, w=512, rgb=True):
    """
    génération d'images ecg où les 12 dérivations sont superposées sur la même figure.
    
    args:
        tracings (torch.tensor): tenseur des signaux de forme (batch, channels, time).
        lengths (torch.tensor, optional): longueurs réelles des signaux pour la gestion du padding.
        h (int): hauteur de l'image.
        w (int): largeur de l'image.
        segment_size (int): nombre de points temporels par fenêtre.
        scale_y (float): facteur d'échelle vertical pour l'amplitude.
        rgb (bool): affichage en couleurs distinctes si vrai, monochrome si faux.

    returns:
        tuple: (tenseur d'images [b, s, c, h, w], nombre de fenêtres valides par patient [b])
    """
    b, c, time_steps = tracings.shape

    base_ref = 512.0

    # Automatisation des ratios optimaux
    segment_size = int(w * (1000.0 / base_ref))

    # Amplitude : (2.8/512) * h
    scale_y = h * (2.7 / base_ref)

    # ajout d'un padding temporel si la taille totale n'est pas un multiple exact du segment
    pad_len = (segment_size - (time_steps % segment_size)) % segment_size
    if pad_len > 0:
        tracings = F.pad(tracings, (0, pad_len), mode='constant', value=0.0)

    segments = tracings.unfold(2, segment_size, segment_size)
    b, c, s, l = segments.shape

    # calcul du nombre d'images valides par patient
    if lengths is not None:
        num_windows = torch.ceil(lengths.float() / segment_size).long()
    else:
        num_windows = torch.full((b,), s, dtype=torch.long)

    offsets = np.linspace(h * 0.05, h * 0.95, 12).astype(np.int32)
    x_coords = np.linspace(0, w - 1, l).astype(np.int32)

    # allocation mémoire (fond blanc par défaut)
    num_channels = 3 if rgb else 1
    output_np = np.full((b, s, h, w, num_channels), 255, dtype=np.uint8)

    segments_np = segments.numpy()

    for batch_idx in range(b):
        valid_n = num_windows[batch_idx].item()
        for n in range(s):
            # exclusion stricte des zones de padding pour économiser le processeur
            if n >= valid_n:
                continue

            img_slice = output_np[batch_idx, n]
            for i in range(12):
                sig = segments_np[batch_idx, i, n, :]
                y_coords = (offsets[i] - sig * scale_y).astype(np.int32)
                np.clip(y_coords, 0, h - 1, out=y_coords)

                pts = np.column_stack((x_coords, y_coords)).reshape((-1, 1, 2))

                color = distinct_colors[i] if rgb else 0
                cv2.polylines(img_slice, [pts], False, color, 2, cv2.LINE_AA)

    final_tensor = torch.from_numpy(output_np).permute(0, 1, 4, 2, 3).contiguous()

    if not rgb:
        final_tensor = final_tensor.expand(-1, -1, 3, -1, -1).clone()

    return final_tensor, num_windows


def create_image_12leads_perchan(tracings, lengths=None, h=512, w=512):
    """
    génération d'images ecg optimisée où chaque dérivation possède son propre canal visuel.

    args:
        tracings (torch.tensor): tenseur des signaux bruts.
        lengths (torch.tensor, optional): longueurs réelles des signaux.
        h (int): hauteur allouée à chaque canal individuel.
        w (int): largeur de l'image complète.
        segment_size (int): nombre de points par fenêtre.

    returns:
        tuple: (tenseur d'images [b, s, 12, h, w], nombre de fenêtres valides par patient [b])
    """
    batch_size, channels, time_steps = tracings.shape

    base_ref = 512.0

    ratio_segment = 4000.0 / base_ref
    segment_size = int(w * ratio_segment)

    # alignement de la dimension temporelle
    pad_len = (segment_size - (time_steps % segment_size)) % segment_size
    if pad_len > 0:
        tracings = F.pad(tracings, (0, pad_len), mode='constant', value=0.0)

    segments = tracings.unfold(2, segment_size, segment_size)
    segments_np = segments.transpose(1, 2).numpy()

    batch_size, num_segments, channels, seq_len = segments_np.shape

    # identification du nombre d'images non-vides
    if lengths is not None:
        num_windows = torch.ceil(lengths.float() / segment_size).long()
    else:
        num_windows = torch.full((batch_size,), num_segments, dtype=torch.long)

    fill_ratio = 0.5 - 0.1

    sig_clean = segments_np
    max_amp = np.max(np.abs(sig_clean), axis=(1, 3), keepdims=True)
    scale_y_dynamic = (h * fill_ratio) / (max_amp + 1e-6)

    shift = 4
    mult = 16

    # calculs vectorisés des coordonnées
    y_coords_float = (h / 2.0) - (sig_clean * scale_y_dynamic)
    np.clip(y_coords_float, 0, h - 1, out=y_coords_float) 
    y_coords = (y_coords_float * mult).astype(np.int32)

    total_images = batch_size * num_segments
    y_coords_flat = y_coords.reshape(total_images, 12, seq_len)

    offsets = (np.arange(12) * h * mult).reshape(1, 12, 1)
    y_coords_offset = y_coords_flat + offsets

    x_coords = np.linspace(0, (w - 1) * mult, seq_len).astype(np.int32)

    pts_all = np.empty((total_images, 12, seq_len, 1, 2), dtype=np.int32)
    pts_all[..., 0, 0] = x_coords
    pts_all[..., 0, 1] = y_coords_offset

    # allocation mémoire
    output_images_np = np.zeros((batch_size, num_segments, 12, h, w), dtype=np.uint8)
    canvas_view = output_images_np.reshape(total_images, 12 * h, w)

    for j in range(total_images):
        b = j // num_segments
        n = j % num_segments

        # évitement du rendu pour les fenêtres composées uniquement de padding
        if n >= num_windows[b].item():
            continue

        cv2.polylines(canvas_view[j], list(pts_all[j]), isClosed=False, color=255, 
                      thickness=1, lineType=cv2.LINE_8, shift=shift)

    return torch.from_numpy(output_images_np), num_windows



if __name__ == "__main__":
    file_path = "../../../../output/normalize_data/georgia.hdf5"
    output_path = "../../../../output/img/"
    os.makedirs(output_path, exist_ok=True)
    h = 512
    w = 512

    with h5py.File(file_path, 'r') as f:
        tracings = torch.from_numpy(f['tracings'][:10])

    """
    images_tensor, _ = create_image_12leads_together(tracings, None, h, w)

    # Fusion des dimensions du lot et des segments pour l'exportation
    # Regroupement sequentiel des images generees
    flat_images = images_tensor.view(-1, 3, h, w)

    # Enregistrement d'un echantillon pour le controle visuel
    for i in range(min(10, flat_images.size(0))):
        img_float = flat_images[i].to(torch.float32) / 255.0
        # Enregistrement direct au format image standard
        vutils.save_image(img_float, f"{output_path}/check_dino_1channels_{i}.png")
    """

    images_tensor, _ = create_image_12leads_perchan(tracings, None, h, w)

    print(f"Format du tenseur genere : {images_tensor.shape}")

    flat_images = images_tensor.view(-1, 12, h, w)

    for i in range(min(5, flat_images.size(0))):
        grid_input = flat_images[i].unsqueeze(1).float() / 255.0
        vutils.save_image(grid_input, f"../../../../output/img/check_dino_12channels_{i}.png", nrow=4, normalize=False)

    print(f"Verification terminee : echantillons sauvegardes depuis le tenseur de taille {images_tensor.shape}")
