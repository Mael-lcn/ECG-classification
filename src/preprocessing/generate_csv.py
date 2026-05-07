import csv
from pathlib import Path
import argparse, os
from tqdm import tqdm
import multiprocessing
import time



SNOMED_TO_CLASS = {
    "164889003": "AF",
    "164890007": "AFL",
    "6374002": "BBB",
    "426627000": "Brady",
    "270492004": "IAVB",
    "59118001": "RBBB",
    "713427006": "CRBBB",
    "713426002": "IRBBB",
    "164909002": "LBBB",
    "733534002": "CLBBB",
    "39732003": "LAD",
    "445118002": "LAnFB",
    "164947007": "LPR",
    "251146004": "LQRSV",
    "111975006":  "LQT",
    "698252002": "NSIVCB",
    "426783006": "NSR",
    "284470004": "PAC",
    "10370003": "PR",
    "365413008": "PRWP",
    "427172004": "PVC",
    "164917005": "QAb",
    "47665007": "RAD",
    "427393009": "SA",
    "426177001": "SB",
    "427084000": "STach",
    "63593006": "SVPB",
    "164934002": "TAb",
    "59931005": "TInv",
    "17338001": "VPB",
}


FINAL_CLASSES = [
    "AF","AFL","BBB","Brady","LBBB","RBBB","1dAVb",
    "LAD","LAnFB","LPR","LQRSV", "LQT", "NSIVCB","NSR","PAC", 
    "PR","PRWP","PVC","QAb","RAD","SA","SB","ST",
    "SVPB","TAb","TInv","VPB"
]

def parse_hea_file(hea_file):
    """Parse WFDB header file to extract exam_id, age, sex, and Dx codes"""
    hea_file = Path(hea_file)
    dataset_name = hea_file.parents[1].name

    exam_id = hea_file.stem
    age = None
    is_male = None
    dx_codes = []
    freq = None

    with open(hea_file, "r") as f:
        for line in f:
            line = line.strip()

            # --- FIRST LINE: sampling frequency ---
            if freq is None and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        freq = float(parts[2])
                    except:
                        freq = None
                continue

            # --- METADATA LINES ---
            if line.startswith("# Age:"):
                try:
                    age = float(line.split(":")[1].strip())
                except:
                    age = None
            elif line.startswith("# Sex:"):
                sex = line.split(":")[1].strip().lower()
                is_male = True if sex == "male" else False
            elif line.startswith("# Dx:"):
                dx_codes = [c.strip() for c in line.split(":")[1].split(",")]

    dx = {SNOMED_TO_CLASS[c] for c in dx_codes if c in SNOMED_TO_CLASS}

    labels = {k: False for k in FINAL_CLASSES}

    labels["1dAVb"] = "IAVB" in dx
    labels["ST"] = "STach" in dx

    labels["RBBB"] = any(x in dx for x in ["RBBB","CRBBB","IRBBB"])
    labels["LBBB"] = any(x in dx for x in ["LBBB","CLBBB"])

    has_sb = "SB" in dx
    has_brady = "Brady" in dx

    # Sinus bradycardia -> SB = 1, Brady = 0
    if has_sb:
        labels["SB"] = True
        labels["Brady"] = False

    # Non-sinus bradycardia -> Brady = 1
    elif has_brady:
        labels["SB"] = False
        labels["Brady"] = True

    # No bradycardia
    else:
        labels["SB"] = False
        labels["Brady"] = False

    for k in ["AF", "AFL", "BBB", "LAD","LAnFB","LPR","LQRSV", "LQT", "NSIVCB","NSR",
              "PAC","PR","PRWP","PVC","QAb","RAD","SA",
              "SVPB","TAb","TInv","VPB"]:
        labels[k] = k in dx

    row = {
        "exam_id": exam_id,
        "age": age,
        "is_male": is_male,
        "nn_predicted_age": "",
        **labels,
        "death": "",
        "timey": "",
        "normal_ecg": "",
        "frequences": freq,
        "trace_file": f"{dataset_name}.hdf5"
    }

    return dataset_name, row


def run(args):
    start_time = time.time()

    train_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    hea_files = sorted(train_root.rglob("*.hea"))

    rows_per_dataset = {}

    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for dataset_name, row in tqdm(
            pool.imap(parse_hea_file, hea_files),
            total=len(hea_files),
            desc="Parsing .hea files"
        ):
            rows_per_dataset.setdefault(dataset_name, []).append(row)

    global_patient_id = 1

    # Generate CSV per dataset
    for dataset_name, rows in rows_per_dataset.items():
        out_csv = output_root / f"{dataset_name}.csv"

        for row in rows:
            row["patient_id"] = global_patient_id
            global_patient_id += 1

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "exam_id", "age", "is_male", "nn_predicted_age",
                    *FINAL_CLASSES,
                    "patient_id","death","timey","normal_ecg","frequences","trace_file"
                ]
            )
            writer.writeheader()
            writer.writerows(rows)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Completed in {minutes} minutes and {seconds} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="../../../data/physioNet")
    parser.add_argument("-o", "--output", type=str, default="../../../output/dataset1/")
    parser.add_argument("-w", "--workers", type=int, default=multiprocessing.cpu_count()-1)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    run(args)

if __name__ == "__main__":
    main()
