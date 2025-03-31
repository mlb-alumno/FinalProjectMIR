import os

import librosa
import soundfile as sf
from audiomentations import (
    AddGaussianNoise,
    Aliasing,
    BitCrush,
    Compose,
    Mp3Compression,
    RoomSimulator,
)
from tqdm import tqdm

# Adjust these values to taste:
augment_pipeline = Compose(
    [
        AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.0005, p=0.5),
        Aliasing(p=0.25),
        BitCrush(p=0.25),
        Mp3Compression(p=0.25),
        RoomSimulator(p=0.5),
    ]
)

input_folder = "ADDRESS_TO_YOUR_INPUT_FOLDER"
output_folder = "ADDRESS_TO_YOUR_OUTPUT_FOLDER"
os.makedirs(output_folder, exist_ok=True)

for file_name in tqdm(os.listdir(input_folder)):
    if file_name.lower().endswith(".wav"):
        file_path = os.path.join(input_folder, file_name)

        audio, sr = librosa.load(file_path, sr=16000, mono=False)
        # Apply Gaussian noise
        augmented_audio = augment_pipeline(samples=audio, sample_rate=sr)

        # If shape is (channels, samples), transpose to (samples, channels)
        if augmented_audio.ndim == 2:
            augmented_audio = augmented_audio.T

        out_file = os.path.join(output_folder, file_name)
        sf.write(out_file, augmented_audio, sr, subtype="PCM_16")
