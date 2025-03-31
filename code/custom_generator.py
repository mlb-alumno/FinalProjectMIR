import numpy as np
import tensorflow as tf


class CustomGenerator:
    def __init__(
        self,
        mux,
        input_shape,
        output_shape,
        structured=False,
        dtype=tf.float32,
    ):
        self.mux = mux
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.structured = structured  # Add this flag
        self.dtype = dtype

    def __iter__(self):
        return self.generator()

    def generator(self):
        for item in self.mux.iterate():
            # Convert Python lists to NumPy arrays (TensorFlow needs this)
            x = np.array(item["cqt_mag"], dtype=np.float32)

            if self.structured:
                # Return all necessary outputs for structured training
                y_tag = np.array(item["chord_tag_chord"], dtype=np.int32)
                y_pitch = np.array(
                    item["chord_struct_pitch"], dtype=np.float32
                )
                y_root = np.array(item["chord_struct_root"], dtype=np.int32)
                y_bass = np.array(item["chord_struct_bass"], dtype=np.int32)
                yield x, (y_tag, y_pitch, y_root, y_bass)
            else:
                # Return only chord tags for non-structured training
                y = np.array(item["chord_tag_chord"], dtype=np.int32)
                yield x, y

    def as_tf_dataset(self, batch_size):
        # This function would need to be updated too but isn't used in the fit() call
        pass
