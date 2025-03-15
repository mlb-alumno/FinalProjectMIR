import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Create a CSV with file names (without extensions) from a specified folder."
    )
    parser.add_argument("--folder", type=str, required=True,
                        help="Path to the folder to scan for files.")
    parser.add_argument("--output", type=str, default="file_names.csv",
                        help="Output CSV file name (default: file_names.csv).")
    args = parser.parse_args()

    folder = args.folder
    file_names = []

    # Iterate through the files in the folder
    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        if os.path.isfile(full_path):
            # Split the file name and extension, keep the name only
            name_without_ext, _ = os.path.splitext(file)
            file_names.append(name_without_ext)
    
    # Create a DataFrame with a single column
    df = pd.DataFrame(file_names, columns=["filename"])
    df.to_csv(args.output, index=False)
    print(f"CSV saved as {args.output}")

if __name__ == "__main__":
    main()
