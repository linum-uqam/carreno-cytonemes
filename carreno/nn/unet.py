# -*- coding: utf-8 -*-
import tensorflow as tf


def two_conv(input, n_feat=32, conv_size=[3,3]):
    """
    2 convolution layers to add with batch normalisation
    Parameters
    ----------
    input : tf.keras.engine.keras_tensor.KerasTensor
        input layer to pick up from
    n_feat : int
        number of features for convolutions
    conv_size : [int, int]
        shape of convolution kernels
    Returns
    -------
    __ " tf.keras.engine.keras_tensor.KerasTensor
        last keras layer output
    """
    conv1 = tf.keras.layers.Conv2D(n_feat, conv_size, padding="same")(input)
    norm1 = tf.keras.layers.BatchNormalization()(conv1)
    acti1 = tf.keras.layers.LeakyReLU()(norm1)
    conv2 = tf.keras.layers.Conv2D(n_feat, conv_size, padding="same")(acti1)
    norm2 = tf.keras.layers.BatchNormalization()(conv2)
    acti2 = tf.keras.layers.LeakyReLU()(norm2)
    return acti2


def encoder_block(input, n_feat=32, conv_size=[3,3], pool_size=[2,2]):
    """
    Encoder block for an UNet
    Parameters
    ----------
    input : tf.keras.engine.keras_tensor.KerasTensor
        previous layer to start from
    n_feat : int
        Number of features for this block
    conv_size : [int, int]
        Kernel size for convolutions
    pool_size : [int, int]
        Kernel size for down sampling
    Returns
    -------
    skip : tf.keras.engine.keras_tensor.KerasTensor
        skip layers to concatenate with decoder
    down_sample : tf.keras.engine.keras_tensor.KerasTensor
        encoder block output
    """
    skip = two_conv(input, n_feat, conv_size)
    down_sample = tf.keras.layers.MaxPooling2D(pool_size)(skip)
    return skip, down_sample


def decoder_block(input, skip, n_feat=32, conv_size=[3,3], pool_size=[2,2], stride=[2,2]):
    """
    Decoder block for an UNet
    Parameters
    ----------
    input : tf.keras.engine.keras_tensor.KerasTensor
        previous layer to start from
    n_feat : int
        Numbder of features for this block
    conv_size : [int, int]
        Kernel size for convolutions
    pool_size : [int, int]
        Kernel size for upsampling
    Returns
    -------
    out : tf.keras.engine.keras_tensor.KerasTensor
        decoder block output
    """
    upsample = tf.keras.layers.Conv2DTranspose(n_feat, pool_size, stride)(input)
    skip_concatenate = tf.keras.layers.Concatenate()([upsample, skip])
    out = two_conv(skip_concatenate, n_feat, conv_size)
    return out


def unet2D(shape, depth=3, n_classes=3, n_feat=32):
    """
    Create a 2D UNet architecture
    Parameters
    ----------
    shape : (int, int, int)
        Image shape. Even if grayscale, we must have a color channel
    depth : int
        UNet number of levels
    n_class : int
        Number of unique labels
    n_feat : int
        Number of features for the first encoder block (will increase and decrease according to UNet architecture)
    Returns
    -------
    model : tf.keras.Model
        Keras model waiting to be compiled for training
    """
    kernel_size_sampling = (2, 2, 2)  # kernel size for sampling operation is smaller on first axis since it's very short
    kernel_size_conv = (3, 3, 3)

    inp = tf.keras.layers.Input(shape)

    # encoder
    skip_layer = []
    current_layer = inp
    for i in range(depth-1):
        skip, current_layer = encoder_block(current_layer,
                                            n_feat * (2 ** i),
                                            [3,3],
                                            [2,2])
        skip_layer.append(skip)

    # middle
    current_layer = two_conv(current_layer,
                             n_feat * (2 ** (depth-1)))
    
    # decoder
    for i in range(depth-2, -1, -1):
        current_layer = decoder_block(current_layer,
                                      skip_layer[i],
                                      n_feat * (2 ** i),
                                      [3,3],
                                      [2,2])
        
    out = tf.keras.layers.Conv2D(filters=n_classes,
                                 kernel_size=1,
                                 padding="same",
                                 activation="softmax")(current_layer)

    model = tf.keras.Model(inp, out)
    
    # try doing out.summary() to make sure everything is alright
    return model


def conv_transfer_learning(model, conv_layers):
    pass