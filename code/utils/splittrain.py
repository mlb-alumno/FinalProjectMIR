import os
import shutil
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Copy .wav files based on CSV criteria"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing subfolders with .wav files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help='Output directory where "train" and "test" folders will be created',
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the CSV file containing file information",
    )
    args = parser.parse_args()

    source_folder = args.root
    output_dir = args.output
    csv_file = args.csv

    # Create destination folders for train and test inside the output directory.
    train_folder = os.path.join(output_dir, "train")
    test_folder = os.path.join(output_dir, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Read the CSV file.
    df = pd.read_csv(csv_file)

    # Process each row in the CSV.
    for index, row in df.iterrows():
        # Only process rows where the 'type' column equals "chord"
        type_value = str(row["type"]).strip().lower()
        if type_value != "chord":
            continue

        file_name = str(row["name"]).strip()
        # Ensure the file name ends with '.wav'
        if not file_name.lower().endswith(".wav"):
            file_name += ".wav"

        # Determine the destination folder based on the 'split' column.
        split_value = str(row["split"]).strip().lower()
        if split_value == "train":
            dest_folder = train_folder
        elif split_value == "test":
            dest_folder = test_folder
        else:
            # Skip rows with split values other than 'train' or 'test'
            continue

        # Recursively search for the file in the source_folder.
        file_found = False
        for root, dirs, files in os.walk(source_folder):
            if file_name in files:
                src_path = os.path.join(root, file_name)
                dest_path = os.path.join(dest_folder, file_name)
                shutil.copy(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")
                file_found = True
                break

        if not file_found:
            print(f"File {file_name} not found in {source_folder}.")

if __name__ == "__main__":
    main()
