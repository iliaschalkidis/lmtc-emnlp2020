import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D


class GlobalMeanPooling1D(GlobalAveragePooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(GlobalMeanPooling1D, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, dtype=x.dtype), axis=-1)
            x = x * mask
            return tf.reduce_sum(x, axis=1) / tf.reduce_sum(mask, axis=1)
        else:
            return tf.reduce_mean(x, axis=1)

    def compute_output_shape(self, input_shape):
        # remove temporal dimension
        return input_shape[0], input_shape[2]
