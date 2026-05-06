"""
Computes per-class pos_weight = (N - P) / P across ALL normalized CSV files,
handling the two subdataset formats:
  - Full format  : 27 classes (PhysioNet)
  - Partial format: 7 classes (15Percent: 1dAVb, RBBB, LBBB, SB, ST, AF + NSR)

The resulting pos_weight tensor is saved as:
  ressources/pos_weight.pt ready to be loaded in train.py
"""

import os
import json
import argparse
import glob
import numpy as np
import pandas as pd
import torch



def compute_pos_weight(data_dir: str, all_classes: list, output_dir: str):
    """
    Scans all CSV files under data_dir, accumulates positive and total counts
    per class across all subdatasets (regardless of their column subset),
    then computes pos_weight = (N - P) / P for each class.

    Classes absent from a given CSV are simply skipped for that file —
    they do NOT contribute zeros, which would unfairly inflate negative counts.
    """
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    print(f"Found {len(csv_files)} CSV file(s):\n  " + "\n  ".join(csv_files))

    # Accumulators: total ECG count and positive count per class
    total_counts = {cls: 0 for cls in all_classes}
    pos_counts   = {cls: 0 for cls in all_classes}

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        n_rows = len(df)
        print(f"\n[{os.path.basename(csv_path)}]  {n_rows} ECGs")

        # Identify which of our 27 classes are present in this CSV
        available = [cls for cls in all_classes if cls in df.columns]
        missing   = [cls for cls in all_classes if cls not in df.columns]

        print(f"  Classes present : {len(available)}")
        if missing:
            print(f"  Classes absent  : {missing}  → skipped for this file")

        for cls in available:
            # Cast booleans / strings to int safely
            col = df[cls].map(lambda v: 1 if str(v).strip().lower() == "true" else 0)
            p = int(col.sum())
            pos_counts[cls]   += p
            total_counts[cls] += n_rows

    # Print summary table 
    print("\n" + "=" * 65)
    print(f"{'Class':<12} {'Total ECGs':>12} {'Positives':>12} {'Negatives':>12} {'pos_weight':>12}")
    print("=" * 65)

    pos_weight_values = []
    for cls in all_classes:
        total = total_counts[cls]
        p     = pos_counts[cls]
        n     = total - p

        if total == 0:
            # Class never appeared in any CSV — assign weight 1.0 (neutral)
            w = 1.0
            print(f"{cls:<12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'1.0 (default)':>12}")
        elif p == 0:
            # Class present in some CSVs but zero positives — very rare edge case
            # Assign a high weight equal to total (entire dataset is negative)
            w = float(total)
            print(f"{cls:<12} {total:>12,} {p:>12,} {n:>12,} {w:>12.2f}  NO POSITIVES!")
        else:
            w = n / p
            print(f"{cls:<12} {total:>12,} {p:>12,} {n:>12,} {w:>12.2f}")

        pos_weight_values.append(w)

    print("=" * 65)

    # Save as .pt tensor
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "pos_weight.pt")

    pos_weight_tensor = torch.tensor(pos_weight_values, dtype=torch.float32)
    torch.save(pos_weight_tensor, save_path)

    print(f"\npos_weight tensor saved → {save_path}")
    print(f"  Shape : {pos_weight_tensor.shape}")
    print(f"  Min   : {pos_weight_tensor.min():.4f}  |  Max : {pos_weight_tensor.max():.4f}")

    # Save a human-readable JSON for reference
    json_path = os.path.join(output_dir, "pos_weight.json")
    with open(json_path, "w") as f:
        json.dump({cls: round(w, 6) for cls, w in zip(all_classes, pos_weight_values)}, f, indent=2)
    print(f"Human-readable weights: {json_path}")

    return pos_weight_tensor


def main():
    parser = argparse.ArgumentParser(description="Compute pos_weight for BCEWithLogitsLoss")
    parser.add_argument("--data_dir",     type=str, required=True, default="../output/normalize_data",
                        help="Root folder containing all normalized CSV files (searched recursively)")
    parser.add_argument("--classes_json", type=str, required=True, default="/ressources/final_class.json",
                        help="Path to final_classes.json")
    parser.add_argument("--output_dir",   type=str, required=True, default="/ressources",
                        help="Where to save pos_weight.pt and pos_weight.json")
    args = parser.parse_args()

    # Load class list
    with open(args.classes_json) as f:
        classes = json.load(f)
    print(f"Loaded {len(classes)} classes from {args.classes_json}")


    compute_pos_weight(args.data_dir, classes, args.output_dir)


if __name__ == "__main__":
    main()
