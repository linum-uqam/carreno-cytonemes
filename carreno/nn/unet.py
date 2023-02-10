# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

import carreno.nn.layers


def UNet(shape, n_class=3, depth=3, n_feat=32, backbone=None):
    """
    Create a UNet architecture
    Parameters
    ----------
    shape : [int] * 3 or 4
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
    gray_input = shape[-1] == 1
    depth = depth
    ndim = len(shape) - 1
    kernel_size_conv =     [3] * ndim  # kernel size for sampling operation is smaller on first axis since it's very short
    kernel_size_sampling = [2] * ndim
    layers = carreno.nn.layers.layers(ndim)


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
        conv1 = layers.ConvXD(n_feat,
                              kernel_size_conv,
                              padding="same")(input)
        norm1 = layers.BatchNormalization()(conv1)
        acti1 = layers.LeakyReLU()(norm1)
        conv2 = layers.ConvXD(n_feat,
                              kernel_size_conv,
                              padding="same")(acti1)
        norm2 = layers.BatchNormalization()(conv2)
        acti2 = layers.LeakyReLU()(norm2)
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
        down_sample = layers.MaxPoolingXD(kernel_size_sampling)(skip)
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
        upsample = layers.ConvXDTranspose(n_feat,
                                          kernel_size_sampling,
                                          kernel_size_sampling)(input)
        skip_concatenate = layers.Concatenate()([upsample, skip])
        out = two_conv(skip_concatenate, n_feat)
        return out

    
    input = layers.Input(shape)
    
    
    skip_layer = []
    current_layer = input

    if backbone is None:
        # encoder
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
    else:
        if "vgg16":
            vgg16 = tf.keras.applications.VGG16(include_top=False,
                                                weights='imagenet',
                                                input_shape=list(shape[:-1])+[3])

            # encoder (including middle block)
            if gray_input:
                """
                I would use the custom initializer to set weights/bias without passing through the model, but we can't load the model
                without the custom init then... Here is what it looked like :
                
                # w, b = vgg16.get_layer('block1_conv1').get_weights()
                # nw = np.expand_dims(w.mean(axis=2), axis=2)  # color channels avg
                # current_layer = tf.keras.layers.Conv2D(64, 3,
                #                                        padding="same",
                #                                        kernel_initializer=lambda shape,dtype:nw,
                #                                        bias_initializer=lambda shape,dtype:b)(current_layer)

                I prefer avoiding saving these lambda functions by setting them later when we have the model even if it's messier.
                """
                current_layer = tf.keras.layers.Conv2D(64, 3,
                                                       padding="same")(current_layer)
            else:
                # use rgb weights and bias
                current_layer = vgg16.layers[1](current_layer)

            for i in range(2, len(vgg16.layers)-1):  # we skip last layer since it's pooling with no weights
                if "pool" in vgg16.layers[i].name:
                    # should be the last convolution layer before pooling
                    skip_layer.append(current_layer)
                current_layer = vgg16.layers[i](current_layer)
            
            # decoder
            current_layer = decoder_block(current_layer,
                                          skip_layer[3],
                                          512)
            for i in range(2, -1, -1):
                current_layer = decoder_block(current_layer,
                                              skip_layer[i],
                                              64 * (2 ** i))
        else:
            raise Exception("Error : " + backbone + " is not supported!") 

    output = None
    if n_class == 2:
        # sigmoid activation
        output = layers.ConvXD(filters=n_class,
                               kernel_size=1,
                               padding="same",
                               activation="sigmoid")(current_layer)
    else:
        # multiclass activation
        output = layers.ConvXD(filters=n_class,
                               kernel_size=1,
                               padding="same",
                               activation="softmax")(current_layer)
    
    model = tf.keras.Model(input, output)

    if backbone and gray_input:
        # refer to backbone explanation above for why I'm setting the weights so late
        if backbone == "vgg16":
            w, b = vgg16.get_layer('block1_conv1').get_weights()
            nw = np.expand_dims(w.mean(axis=2), axis=2)  # color channels avg
            model.layers[1].set_weights([nw, b])

    model.depth = depth
    model.backbone = backbone
    model.encoder_learning = True
    
    def switch_encoder_status(model):
        if model.encoder_learning:
            # freeze encoder TODO
            model.encoder_learning = False
        else:
            # unfreeze encoder TODO
            model.encoder_learning = True

    model.switch_encoder_status = switch_encoder_status

    return model


def add_backbone(unet, backbone, pretrained=True):
    is_3d = unet.layers
    current_layer = unet.layers[0]
    skip_layers = []

    if backbone == "vgg16":
        if unet.depth != 5:
            raise Exception("Error : UNet architecture incompatible, depth must be 5.")
        
        # encoder
        current_layer = ...

    # decoder
    current_layer = ...

    return tf.keras.Model(unet.layer[0], current_layer)