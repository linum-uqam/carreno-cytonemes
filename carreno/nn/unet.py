# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def UNet(shape, n_class=3, depth=3, n_feat=32):
    """
    Create a UNet architecture
    Parameters
    ----------
    shape : (int, int, int)
        Image shape. Even if grayscale, we must have a color channel
    n_class : int
        Number of unique labels
    depth : int
        UNet number of levels (nb of encoder block + 1)
    n_feat : int
        Number of features for the first encoder block (will increase and decrease according to UNet architecture)
    Returns
    -------
    model : tf.keras.Model
        Keras model waiting to be compiled for training
    """
    depth = depth
    ndim = len(shape) - 1
    kernel_size_conv =     [3] * ndim  # kernel size for sampling operation is smaller on first axis since it's very short
    kernel_size_sampling = [2] * ndim
    conv_layer_f = tf.keras.layers.Conv2D
    pool_layer_f = tf.keras.layers.MaxPooling2D
    transpose_layer_f = tf.keras.layers.Conv2DTranspose
    if ndim == 3:
        conv_layer_f = tf.keras.layers.Conv3D
        pool_layer_f = tf.keras.layers.MaxPooling3D
        transpose_layer_f = tf.keras.layers.Conv3DTranspose


    def two_conv(input, n_feat=32):
        """
        2 convolution layers to add with batch normalisation
        Parameters
        ----------
        input : tf.keras.engine.keras_tensor.KerasTensor
            input layer to pick up from
        n_feat : int
            number of features for convolutions
        Returns
        -------
        __ " tf.keras.engine.keras_tensor.KerasTensor
            last keras layer output
        """
        conv1 = conv_layer_f(n_feat,
                             kernel_size_conv,
                             padding="same")(input)
        norm1 = tf.keras.layers.BatchNormalization()(conv1)
        acti1 = tf.keras.layers.LeakyReLU()(norm1)
        conv2 = conv_layer_f(n_feat,
                             kernel_size_conv,
                             padding="same")(acti1)
        norm2 = tf.keras.layers.BatchNormalization()(conv2)
        acti2 = tf.keras.layers.LeakyReLU()(norm2)
        return acti2


    def encoder_block(input, n_feat=32):
        """
        Encoder block for an UNet
        Parameters
        ----------
        input : tf.keras.engine.keras_tensor.KerasTensor
            previous layer to start from
        n_feat : int
            Number of features for this block
        Returns
        -------
        skip : tf.keras.engine.keras_tensor.KerasTensor
            skip layers to concatenate with decoder
        down_sample : tf.keras.engine.keras_tensor.KerasTensor
            encoder block output
        """
        skip = two_conv(input, n_feat)
        down_sample = pool_layer_f(kernel_size_sampling)(skip)
        return skip, down_sample


    def decoder_block(input, skip, n_feat=32):
        """
        Decoder block for an UNet
        Parameters
        ----------
        input : tf.keras.engine.keras_tensor.KerasTensor
            previous layer to start from
        skip : tf.keras.engine.keras_tensor.KerasTensor
            skip layer to concatenate with decoder block
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
        upsample = transpose_layer_f(n_feat,
                                     kernel_size_sampling,
                                     kernel_size_sampling)(input)
        skip_concatenate = tf.keras.layers.Concatenate()([upsample, skip])
        out = two_conv(skip_concatenate, n_feat)
        return out

    
    input = tf.keras.layers.Input(shape)
    
    # encoder
    skip_layer = []
    current_layer = input
    for i in range(depth-1):
        skip, current_layer = encoder_block(current_layer,
                                            n_feat * (2 ** i))
        skip_layer.append(skip)
    # middle
    current_layer = two_conv(current_layer,
                             n_feat * (2 ** (depth-1)))

    # decoder
    for i in range(depth-2, -1, -1):
        current_layer = decoder_block(current_layer,
                                      skip_layer[i],
                                      n_feat * (2 ** i))
    
    output = conv_layer_f(filters=n_class,
                          kernel_size=1,
                          padding="same",
                          activation="softmax")(current_layer)
    
    model = tf.keras.Model(input, output)

    return model


def get_layer_parent_names(layer):
    """
    Get name of layers directly preceding given layer
    Parameters
    ----------
    layer : tf.keras.layers.Layer
        keras layer
    Returns
    -------
    model : list
        list of parent names
    """
    parents = []
    for inp in layer.input:
        parents.append(inp.name.split('/')[0])
    return parents


def get_layer_parent_i(layer, model):
    """
    Get indexes of layers directly preceding given layer for model
    Parameters
    ----------
    layer : tf.keras.layers.Layer
        keras layer
    model : tf.keras.Model
        keras model
    Returns
    -------
    model : {str : int}
        dict of parents, key is the name of layer and value is index in model layer
    """
    parent_names = get_layer_parent_names(layer)
    parent_i = {}
    layer_names = [l.name for l in model.layers]
    for name in parent_names:
        parent_i[name] = -1
        try:
            parent_i[name] = layer_names.index(name)
        except:
            pass
    
    return parent_i


def weight2D_to_3D(weights, dim):
    """
    Adapts 2D weights to 3D
    Parameters
    ----------
    weights : ndarray
        2D weights to convert
    dim : int
        how many values over the new axis for 3D
    Returns
    -------
    weight3D : ndarray
        avg 2D weights over new axis
    """
    weight3D = np.zeros([dim] + list(weights.shape), dtype=weights.dtype)
    avg_w = weights / dim
    
    # add avg weights over third axis
    weight3D[:] = avg_w
    
    return weight3D


def unet2D_to_unet3D(unet2D, shape):
    """
    Go over every 2D layers and convert 2D weights to 3D if available
    Parameters
    ----------
    unet2D : tf.keras.Model
        2D UNet to convert
    shape : list
        input shape for 3D input
    Returns
    -------
    unet3D : UNet
        3D UNet with 2D weights
    """
    n_class = unet2D.layers[-1].get_weights()[1].shape[0]
    n_feat = unet2D.layers[1].get_weights()[1].shape[0]
    depth = 1
    for i in range(len(unet2D.layers)):
        if 'POOL' in unet2D.layers[i].name.upper():
            depth += 1
    
    unet3D = UNet(shape=shape,
                  depth=depth,
                  n_class=n_class,
                  n_feat=n_feat)
    
    for i in range(len(unet3D.layers)):
        layer2D = unet2D.layers[i]
        layer3D = unet3D.layers[i]
        
        # layer name without default tf int ID (layer_name_ID)
        # do not use, tf.keras first layer instance doesn't end with _#
        #name2D = layer2D.name.rsplit('_', 1)[0]
        
        # must transfer 2D weights to 3D
        if '2d' in layer2D.name:
            try:
                weights2D = layer2D.get_weights()
                weights3D = layer3D.get_weights()
                
                # assume kernel size is uniform
                third_axis_dim = weights2D[0].shape[0]
                weights2D_avg = weights2D[0] / third_axis_dim
                weights3D[0][:] = weights2D_avg
                
                layer3D.set_weights(weights3D)
            except:
                # probably a pooling operation without weights
                #print('Could not transfer layer', name2D, 'to', name3D)
                pass
    
    return unet3D