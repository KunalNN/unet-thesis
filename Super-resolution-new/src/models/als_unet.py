# src/models/als_unet.py
import tensorflow as tf
from tensorflow.keras.layers import Layer
from .unet import conv_block
from tensorflow.keras.layers import (
    Input, MaxPooling2D, Conv2DTranspose, concatenate, UpSampling2D, Conv2D
)
from tensorflow.keras.models import Model

class Scale(Layer):
    """A scalar multiplier (learnable) applied per-feature-map."""
    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
        )
    def call(self, x):
        return self.alpha * x

def build_als_unet_sr(
    input_shape=(None, None, 3),
    scale=4,
    base_filters=64,
):
    """
    UNet with a learnable scale on each skip-connection.
    """
    inputs = Input(shape=input_shape)
    x = UpSampling2D((scale, scale), interpolation="bilinear")(inputs)

    skips = []
    # Encoder with Scale
    for i in range(4):
        x = conv_block(x, base_filters * (2 ** i))
        s = Scale(name=f"scale_enc_{i}")(x)
        skips.append(s)
        x = MaxPooling2D((2, 2))(s)

    # Bottleneck
    x = conv_block(x, base_filters * (2 ** 4))

    # Decoder with Scale
    for i in reversed(range(4)):
        x = Conv2DTranspose(base_filters * (2 ** i), 2, strides=2, padding="same")(x)
        # apply scale on skip before concatenation
        s = Scale(name=f"scale_dec_{i}")(skips[i])
        x = concatenate([s, x])
        x = conv_block(x, base_filters * (2 ** i))

    outputs = Conv2D(3, 1, padding="same", activation="sigmoid")(x)
    return Model(inputs, outputs, name="ALS_UNet_SR")
