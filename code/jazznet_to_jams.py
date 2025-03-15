import os
import argparse
import pandas as pd
import jams
import re

def main():
    parser = argparse.ArgumentParser(
        description="Create JAMS files with chord annotations from a CSV file using JAMS standards."
    )
    parser.add_argument("--working", type=str, required=True,
                        help="Base folder path where 'train' and 'test' subdirectories will be created.")
    parser.add_argument("--csv_file", type=str, default="small.csv",
                        help="Path to the CSV file (default: small.csv).")
    args = parser.parse_args()

    # Create target directories for train and test splits
    train_folder = os.path.join(args.working, "jams/train")
    test_folder = os.path.join(args.working, "jams/test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)


    # Mapping for mode conversion: CSV mode -> JAMS standard chord quality
    mode_mapping = {
        "aug-chord": "aug",
        "dim-chord": "dim",
        "dim7-chord": "dim7",
        "maj-chord": "maj",
        "maj6-chord": "maj6",
        "maj7-chord": "maj7",
        "min-chord": "min",
        "min6-chord": "min6",
        "min7-chord": "min7",
        "min-chord": "min",
        "min7b5-chord": "hdim7",
        "perf5-chord": "5",
        "seventh-chord": "7",
        "sus2-chord": "sus2",
        "sus-4-chord": "sus4"
    }

    # Load CSV data
    df = pd.read_csv(args.csv_file)
    # Process only rows with type 'chord'
    df_chords = df[
    (df["type"] == "chord") & 
    (df["mode"].str.strip().isin(mode_mapping.keys()))
    ]

    # Mapping for inversion conversion: CSV inversion -> jams inversion value
    inversion_mapping = {0: "", 1: "3", 2: "5", 3: "7"}

    for _, row in df_chords.iterrows():
        # Use regex to extract the root note (e.g., C, Ab, Gb) from the beginning of the 'name' column
        match = re.match(r"^([A-G](?:b|#)?)", row["name"])
        if match:
            root_note = match.group(1)
        else:
            root_note = row["name"]  # fallback if regex fails

        # Convert mode using the mapping; if not found, use the original value (or you could skip the row)
        csv_mode = row["mode"].strip()
        converted_mode = mode_mapping.get(csv_mode, csv_mode)

        # Process inversion: try to convert to int; if fails, default to 0
        try:
            inversion_value = int(row["inversion"])
        except ValueError:
            inversion_value = 0

        # Map inversion value to its corresponding string
        inv_str = inversion_mapping.get(inversion_value, "")
        
        # Build chord symbol according to jams standard:
        # If inversion mapping is empty, do not append the slash.
        if inv_str:
            chord_symbol = f"{root_note}:{converted_mode}/{inv_str}"
        else:
            chord_symbol = f"{root_note}:{converted_mode}"

        # Create a new JAMS object and set file metadata
        jam = jams.JAMS()
        jam.file_metadata.artist = "Unknown Artist"
        jam.file_metadata.title = row["name"]
        jam.file_metadata.duration = 3.0

        # Create a chord annotation with the proper namespace and duration
        chord_anno = jams.Annotation(namespace='chord')
        chord_anno.duration = 3.0

        # Append a chord event: start at 0.0 seconds, duration 3.0 seconds, chord value as constructed
        chord_anno.append(time=0.0, duration=3.0, value=chord_symbol)
        jam.annotations.append(chord_anno)

        # Determine output folder based on the 'split' column (default to 'train' if not 'test')
        split_value = row["split"].strip().lower()
        out_folder = test_folder if split_value == "test" else train_folder

        # Save the JAMS file using the CSV's 'name' column as the file name
        jam_filename = os.path.join(out_folder, f"{row['name']}.jams")
        jam.save(jam_filename)
        print(f"Saved: {jam_filename}")

if __name__ == "__main__":
    main()
