import argparse
import os
import random

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Split files in a folder into N train and test sets and save their names to separate CSV files."
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the folder to scan for files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated CSV files.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Base name for the output CSV files.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        required=True,
        help="Number of train/test splits to generate.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion of files to include in the training set (default: 0.8).",
    )
    args = parser.parse_args()

    print("Arguments:")
    print(args)

    folder = args.folder
    output_path = args.output_path
    base_name = args.name
    splits = args.splits
    train_ratio = args.train_ratio
    file_names = []

    # Ensure output path exists
    os.makedirs(output_path, exist_ok=True)

    # Iterate through the files in the folder
    # List of non-desirable files to exclude
    non_desirable_files = []

    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        if os.path.isfile(full_path):
            # Check if the file has a .jams extension
            if file.endswith(".jams"):
                # Split the file name and extension, keep the name only
                name_without_ext, _ = os.path.splitext(file)
                # Exclude non-desirable files
                if name_without_ext not in non_desirable_files:
                    file_names.append(name_without_ext)

    for i in range(splits):
        # Shuffle the file names
        random.shuffle(file_names)  # Randomize the order of files
        train_size = int(len(file_names) * train_ratio)

        # Split into train and test sets
        train_files = file_names[:train_size]
        test_files = file_names[train_size:]

        # Save train set to CSV
        train_df = pd.DataFrame(train_files)
        train_csv_path = os.path.join(
            output_path, f"{base_name}_train0{i}.csv"
        )
        train_df.to_csv(train_csv_path, index=False, header=False)
        print(f"Training CSV saved as {train_csv_path}")

        # Save test set to CSV
        test_df = pd.DataFrame(test_files)
        test_csv_path = os.path.join(output_path, f"{base_name}_test0{i}.csv")
        test_df.to_csv(test_csv_path, index=False, header=False)
        print(f"Test CSV saved as {test_csv_path}")


if __name__ == "__main__":
    main()
