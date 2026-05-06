import pandas as pd
import argparse
import os



def run(args):
    input_csv = args.input
    column_to_split = 'trace_file'
    output_folder = args.output

    try:
        print("1. Chargement du fichier...")
        df = pd.read_csv(input_csv)
        
        df.columns = df.columns.str.strip()

        print(f"   -> {len(df)} lignes chargées.")
        print(f"   -> Colonnes : {df.columns.tolist()}")

        print(f"2. Découpage basé sur : {column_to_split}...")

        for trace_name, group_df in df.groupby(column_to_split):
            # 1. Nettoyage du nom (transtypage en string + strip)
            raw_name = str(trace_name).strip()

            safe_filename = raw_name.replace('.hdf5', '.csv')

            output_path = os.path.join(output_folder, safe_filename)

            # Sauvegarde
            group_df.to_csv(output_path, index=False)
            print(f"   -> Fichier créé : {safe_filename} ({len(group_df)} lignes)")

        os.remove(input_csv)
    except Exception as e:
        print(f"\nERREUR : {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Arguments Dossiers & Fichiers
    parser.add_argument('--input', type=str, default='../../../data/15_prct/exams.csv')
    parser.add_argument('--output', type=str, default='../../../data/15_prct/')

    args = parser.parse_args()

    run(args)
