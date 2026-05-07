import argparse, os
import time
import wfdb
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing



def wfdb_to_hdf5(dataset_dir_out):
    dataset_dir, out_root = dataset_dir_out
    dataset_dir = Path(dataset_dir)
    out_file = out_root / f"{dataset_dir.name}.hdf5"

    # Get all group folders in the dataset
    g_folders = sorted(dataset_dir.glob("g*"))
    exam_ids = []
    signals = []
    max_n_samples = 0

    # Get the max n_samples across all records in the dataset
    for g in g_folders:
        for hea_file in sorted(g.glob("*.hea")):
            record_name = hea_file.stem # Get file name as exam id
            record_path = str(hea_file.with_suffix("")) # Get path without extension

            record = wfdb.rdrecord(record_path)
            # Decode ECG signal in mV, shape=(n_samples, n_leads)
            signal = record.p_signal.astype(np.float32)

            exam_ids.append(record_name)
            signals.append(signal)

            if signal.shape[0] > max_n_samples:
                max_n_samples = signal.shape[0]

    n_ecg = len(signals)
    n_leads = signals[0].shape[1]
    
    # Initialize tracings with zeros
    tracings = np.zeros(
        (n_ecg, max_n_samples, n_leads),
        dtype=np.float32
    )

    # Fill in the tracings with signals of varying lengths
    for i, sig in enumerate(signals):
        n_samples = sig.shape[0]
        tracings[i, :n_samples, :] = sig

    # Create hdf5 file
    with h5py.File(out_file, "w") as f:
        f.create_dataset(
            "exam_id",
            data=np.array(exam_ids, dtype=("S")) # p.s. in hdf5 the strings are stored as bytes  
        )
        f.create_dataset(
            "tracings",
            data=tracings
        )

    return dataset_dir.name


def run(args):
    start_time = time.time()

    train_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_dirs = [
        d for d in sorted(train_root.iterdir())
        if d.is_dir()
    ]

    dataset_items = [(d, out_root) for d in dataset_dirs]

    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        with tqdm(total=len(dataset_items)) as pbar:
            pbar.set_description(f"Converting WFDB datasets to HDF5")
            for result in pool.imap_unordered(wfdb_to_hdf5, dataset_items):
                if result is not None:
                    pbar.update()
                    #tqdm.write(f"Finished {result}")

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
