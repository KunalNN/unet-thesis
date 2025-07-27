# src/models/unet.py
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, UpSampling2D
)
from tensorflow.keras.models import Model

def conv_block(x, filters):
    x = Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def build_unet_sr(
    input_shape=(None, None, 3),
    scale=4,
    base_filters=64,
):
    """
    A simple UNet for super-resolution:
      1) Upsamples input by `scale`.
      2) Runs a U‐Net on the upscaled image.
    """
    inputs = Input(shape=input_shape)
    # 1) initial upsample
    x = UpSampling2D(size=(scale, scale), interpolation="bilinear")(inputs)

    # 2) encoder
    skips = []
    for i in range(4):
        x = conv_block(x, base_filters * (2 ** i))
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)

    # 3) bottleneck
    x = conv_block(x, base_filters * (2 ** 4))

    # 4) decoder
    for i in reversed(range(4)):
        x = Conv2DTranspose(base_filters * (2 ** i), 2, strides=2, padding="same")(x)
        x = concatenate([skips[i], x])
        x = conv_block(x, base_filters * (2 ** i))

    # 5) final reconstruction
    outputs = Conv2D(3, 1, padding="same", activation="sigmoid")(x)

    return Model(inputs, outputs, name="UNet_SR")
