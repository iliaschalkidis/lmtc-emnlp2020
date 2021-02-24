# -*- coding: utf-8 -*-
"""Core Keras layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class SymmetricMasking(Layer):
    """Masks a sequence by using a mask value to skip timesteps based on another sequence.

    For each timestep in the 1st input tensor (dimension #1 in the tensor),
    if all values in the 2nd input tensor at that timestep
    are equal to `mask_value`, then the timestep will be masked (skipped)
    in all downstream layers (as long as they support masking).

    If any downstream layer does not support masking yet receives such
    an input mask, an exception will be raised.

    # Example

    Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
    to be fed to an LSTM layer.
    You want to mask timestep #3 and #5 because you lack data for
    these timesteps. You can:

        - set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
        - insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

    ```python
        model = Sequential()
        model.add(SymmetricMasking(inputs=[1,2] mask_value=0.))
    ```
    """

    def __init__(self, mask_value=0., **kwargs):
        super(SymmetricMasking, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value

    def compute_mask(self, inputs, mask=None):
        if len(inputs[1].shape) == 3:
            output_mask = K.any(K.not_equal(inputs[1], self.mask_value), axis=-1)
        else:
            output_mask = K.not_equal(inputs[1], self.mask_value)
        return output_mask

    def call(self, inputs):
        if len(inputs[1].shape) == 3:
            boolean_mask = K.any(K.not_equal(inputs[1], self.mask_value),
                                 axis=-1, keepdims=True)
        else:
            boolean_mask = K.expand_dims(K.not_equal(inputs[1], self.mask_value))

        return inputs[0] * K.cast(boolean_mask, K.dtype(inputs[0]))

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(SymmetricMasking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Camouflage(Layer):
    """Masks a sequence by using a mask value to skip timesteps based on another sequence.

    For each timestep in the 1st input tensor (dimension #1 in the tensor),
    if all values in the 2nd input tensor at that timestep
    are equal to `mask_value`, then the timestep will be masked (skipped)
    in all downstream layers (as long as they support masking).

    If any downstream layer does not support masking yet receives such
    an input mask, an exception will be raised.

    # Example

    Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
    to be fed to an LSTM layer.
    You want to mask timestep #3 and #5 because you lack data for
    these timesteps. You can:

        - set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
        - insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

    ```python
        model = Sequential()
        model.add(SymmetricMasking(inputs=[1,2] mask_value=0.))
    ```
    """

    def __init__(self, mask_value=0., **kwargs):
        super(Camouflage, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.supports_masking = True

    def call(self, inputs):
        if len(inputs[1].shape) == 3:
            boolean_mask = K.any(K.not_equal(inputs[1], self.mask_value),
                                 axis=-1, keepdims=True)
        else:
            boolean_mask = K.expand_dims(K.not_equal(inputs[1], self.mask_value))

        return inputs[0] * K.cast(boolean_mask, K.dtype(inputs[0]))

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(Camouflage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return None
