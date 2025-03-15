import os
import shutil

# Define input directories
annotation_dir = 'jazznet/jams/test'
audios_dir = 'jazznet/audios_split/test'

# Define output directories (these will be created if they don't exist)
output_annotation_dir = 'jazznet/dataset/jams/test'
output_audios_dir = 'jazznet/dataset/audios/test'

# Create output directories
os.makedirs(output_annotation_dir, exist_ok=True)
os.makedirs(output_audios_dir, exist_ok=True)

# Get lists of files in each directory
annotation_files = os.listdir(annotation_dir)
audios_files = os.listdir(audios_dir)

# Helper function to build a mapping from base name (without extension) to file names
def build_file_dict(files):
    file_map = {}
    for file in files:
        base, ext = os.path.splitext(file)
        file_map.setdefault(base, []).append(file)
    return file_map

ann_dict = build_file_dict(annotation_files)
aud_dict = build_file_dict(audios_files)

# Find common base names between annotation and audios
common_basenames = set(ann_dict.keys()) & set(aud_dict.keys())

# Copy matching files to the new output directories
for basename in common_basenames:
    # Copy annotation files
    for file in ann_dict[basename]:
        src = os.path.join(annotation_dir, file)
        dst = os.path.join(output_annotation_dir, file)
        shutil.copy(src, dst)
    
    # Copy audio files
    for file in aud_dict[basename]:
        src = os.path.join(audios_dir, file)
        dst = os.path.join(output_audios_dir, file)
        shutil.copy(src, dst)

print("Files copied successfully!")
