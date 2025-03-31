import os

import numpy as np
from tensorflow.keras.utils import Sequence


class DataSequence(Sequence):
    def __init__(
        self,
        working,
        tracks,
        sampler,
        batch_size=32,
        augmentation=False,
        weights=None,
        steps=None,
        structured=False,  # To set it to structured
    ):
        self.working = working
        self.tracks = tracks
        self.sampler = sampler
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.weights = weights
        self.steps = steps or 100  # Default number of steps
        self.structured = structured  # Store structured flag

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x = []
        batch_y_tag = []

        # Only used if structured=True
        batch_y_pitch = []
        batch_y_root = []
        batch_y_bass = []

        for _ in range(self.batch_size):
            if self.weights is not None:
                track = np.random.choice(
                    self.tracks, p=self.weights / self.weights.sum()
                )
            else:
                track = np.random.choice(self.tracks)

            base_npz = os.path.join(self.working, "pump", f"{track}.npz")
            data = np.load(base_npz)
            d2 = dict(data)
            data.close()
            sample = next(self.sampler(d2))

            batch_x.append(sample["cqt_mag"])  # Input spectrogram
            batch_y_tag.append(sample["chord_tag_chord"])  # Output labels

            if self.structured:
                batch_y_pitch.append(sample["chord_struct_pitch"])
                batch_y_root.append(sample["chord_struct_root"])
                batch_y_bass.append(sample["chord_struct_bass"])

        batch_x = np.array(batch_x)
        batch_y_tag = np.array(batch_y_tag).squeeze()

        # Ensure it has the correct shape (None, None, 216, 1)
        batch_x = batch_x.reshape(
            batch_x.shape[0], batch_x.shape[2], batch_x.shape[3], 1
        )

        if self.structured:
            batch_y_pitch = np.array(batch_y_pitch)
            batch_y_root = np.array(batch_y_root)
            batch_y_bass = np.array(batch_y_bass)
            return batch_x, (
                batch_y_tag,
                batch_y_pitch,
                batch_y_root,
                batch_y_bass,
            )
        else:
            return batch_x, batch_y_tag
