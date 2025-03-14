from tensorflow.keras.utils import Sequence
import numpy as np 
import tensorflow as tf
import os


class DataSequence(Sequence):
    def __init__(self, working, tracks, sampler, batch_size=32, 
                 augmentation=False, weights=None, steps=None):
        self.working = working
        self.tracks = tracks
        self.sampler = sampler
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.weights = weights
        self.steps = steps or 100  # Default number of steps
        
    def __len__(self):
        return self.steps
        
    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        
        for _ in range(self.batch_size):
            if self.weights is not None:
                track = np.random.choice(self.tracks, p=self.weights/self.weights.sum())
            else:
                track = np.random.choice(self.tracks)

            base_npz = os.path.join(self.working, 'pump', f"{track}.npz")
            data = np.load(base_npz)
            d2 = dict(data)
            data.close()

            sample = next(self.sampler(d2))

            batch_x.append(sample['cqt_mag'])  # Input spectrogram
            batch_y.append(sample['chord_tag_chord'])  # Output labels

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y).squeeze()

        # Ensure it has the correct shape (None, None, 216, 1)
        batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[2], batch_x.shape[3], 1)

        return batch_x, batch_y
