# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import carreno.nn.layers


def encoder_trainable(model, trainable=True):
    """
    Set UNet encoder layers as trainable or not.
    Parameters
    ----------
    model : tf.Keras.Model
        UNet model to set trainability
    trainable : bool
        True to make encoder layers trainable, False for untrainable
    """
    for layer in model.layers:
        if "transpose" in layer.name.lower():
            # decoder is starting
            break
        else:
            layer.trainable = trainable
    
    return


def switch_top(model, activation='softmax'):
    """
    Switch top UNet convolution layer. Useful to experiment different
    activation functions.
    Parameters
    ----------
    model : tf.Keras.Model
        UNet model to set trainability
    trainable : bool
        True to make encoder layers trainable, False for untrainable
    """
    inp               = model.layers[0]
    layers            = carreno.nn.layers.layers(len(inp.input.shape) - 2)
    before_last_layer = model.layers[-2]
    last_layer        = model.layers[-1]
    out               = layers.ConvXD(filters=last_layer.filters,
                                      kernel_size=last_layer.kernel_size,
                                      padding=last_layer.padding,
                                      activation=activation,
                                      name='pred')(before_last_layer.output)
    return tf.keras.Model(inp.input, out)


def UNet(shape, n_class=3, depth=4, n_feat=64, dropout=0.3, batch_norm='after',
         activation=tf.keras.activations.relu, top_activation='softmax', backbone=None, pretrained=True):
    """
    Create a UNet architecture
    Parameters
    ----------
    shape : [int] * 3 or 4
        Image shape. Even if grayscale, we must have a color channel
    n_class : int
        Number of unique labels
    depth : int
        UNet number of levels (nb of encoder block or pooling layers)
    n_feat : int
        Number of features for the first encoder block (will increase and decrease according to UNet architecture)
    dropout : float
        Dropout ratio after convolution activation
    activation : str or keras activation function
        Activation function after convolutions
    top_activation : str or keras activation function
        last convolution layer activation
    backbone : None, str
        UNet backbone, only support "vgg16" atm
    pretrained : bool
        Is the backbone pretrained on imagenet
    Returns
    -------
    model : tf.keras.Model
        Keras model waiting to be compiled for training
    """
    supported_backbone = ["vgg16"]
    depth = depth
    ndim = len(shape) - 1
    kernel_size_conv =     [3] * ndim
    kernel_size_sampling = [2] * ndim
    batch_norm = batch_norm
    gray_input = shape[-1] == 1
    layers = carreno.nn.layers.layers(ndim)
    backbone_model = None
    activation_fn = activation

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
        # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        # See figs_and_nuts answer for layer order
        after = batch_norm == 'after'
        conv1 = layers.ConvXD(n_feat,
                              kernel_size_conv,
                              padding="same")(input)
        
        if after:
            acti1 = layers.Activation(activation_fn)(conv1)
            drop1 = layers.Dropout(dropout)(conv1)
            norm1 = layers.BatchNormalization()(drop1)
        else:
            norm1 = layers.BatchNormalization()(conv1)
            acti1 = layers.Activation(activation_fn)(norm1)
            drop1 = layers.Dropout(dropout)(acti1)

        conv2 = layers.ConvXD(n_feat,
                              kernel_size_conv,
                              padding="same")(norm1 if after else drop1)
        
        if after:
            acti2 = layers.Activation(activation_fn)(conv2)
            drop2 = layers.Dropout(dropout)(conv2)
            norm2 = layers.BatchNormalization()(drop2)
        else:
            norm2 = layers.BatchNormalization()(conv2)
            acti2 = layers.Activation(activation_fn)(norm2)
            drop2 = layers.Dropout(dropout)(acti2)
        
        return norm2 if after else drop2

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
        skip_concatenate = layers.Concatenate()([skip, upsample])
        out = two_conv(skip_concatenate, n_feat)
        return out
    
    input = layers.Input(shape)
    skip_layer = []
    current_layer = input

    if backbone is None:
        # encoder
        for i in range(depth):
            skip, current_layer = encoder_block(current_layer,
                                                    n_feat * (2 ** i))
            skip_layer.append(skip)
        
        # middle
        current_layer = two_conv(current_layer,
                                 n_feat * (2 ** (depth-1)))
        # decoder
        for i in range(depth-1, -1, -1):
            current_layer = decoder_block(current_layer,
                                          skip_layer[i],
                                          n_feat * (2 ** i))
    elif backbone.lower() == supported_backbone[0]:
        if depth != 4 or n_feat != 64:
            print("Warning: UNet architecture with {} backbone will use depth = 4 and n_features = 64.".format(supported_backbone[0]))

        backbone_model = tf.keras.applications.VGG16(include_top=False,
                                                     weights='imagenet' if pretrained else None,
                                                     input_shape=shape[-3:-1]+[3])
        # rm last layer which is a max pooling
        backbone_model = tf.keras.Model(backbone_model.layers[0].input,
                                        backbone_model.layers[-2].output)
        
        if ndim == 3:
            # 2D vgg16 to 3D
            backbone_model = carreno.nn.layers.model2D_to_3D(backbone_model, shape[0])

        if gray_input:
            # instead of averaging the channels, it's recommended to
            # modify input shape to match RGB format. Try to avoid!
            
            """
            I would use the custom initializer to set weights/bias without passing through the model,
            but we can't load the model without the custom init then... Here is what it looked like :
            
            # w, b = backbone_model.layers[1].get_weights()
            # nw = np.expand_dims(w.mean(axis=-1), axis=-1)
            # current_layer = tf.keras.layers.Conv2D(64, 3,
            #                                        padding="same",
            #                                        kernel_initializer=lambda shape,dtype:nw,
            #                                        bias_initializer=lambda shape,dtype:b)(current_layer)

            I prefer avoiding to save these lambda functions by setting weights later when we have the model even if it's messier.
            """
            conv1_layer = backbone_model.layers[1]
            current_layer = layers.ConvXD(filters=conv1_layer.filters,
                                          kernel_size=carreno.nn.layers.__convert_conv_param_for_dim(conv1_layer.kernel_size, ndim),  # add uniform new dim
                                          strides=carreno.nn.layers.__convert_conv_param_for_dim(conv1_layer.strides, ndim),
                                          padding=conv1_layer.padding,
                                          data_format=conv1_layer.data_format,
                                          dilation_rate=carreno.nn.layers.__convert_conv_param_for_dim(conv1_layer.dilation_rate, ndim),
                                          groups=conv1_layer.groups,
                                          activation=conv1_layer.activation,
                                          use_bias=conv1_layer.use_bias,
                                          kernel_initializer=conv1_layer.kernel_initializer,
                                          bias_initializer=conv1_layer.bias_initializer,
                                          kernel_regularizer=conv1_layer.kernel_regularizer,
                                          bias_regularizer=conv1_layer.bias_regularizer,
                                          activity_regularizer=conv1_layer.activity_regularizer,
                                          kernel_constraint=conv1_layer.kernel_constraint,
                                          bias_constraint=conv1_layer.bias_constraint)(current_layer)
        else:
            # use rgb weights and bias
            current_layer = backbone_model.layers[1](current_layer)
        
        for i in range(2, len(backbone_model.layers)):
            if "pool" in backbone_model.layers[i].name:
                # skip layer should be the last convolution layer before pooling
                skip_layer.append(current_layer)
            current_layer = backbone_model.layers[i](current_layer)

        # decoder
        current_layer = decoder_block(current_layer,
                                      skip_layer[3],
                                      512)
        for i in range(2, -1, -1):
            current_layer = decoder_block(current_layer,
                                          skip_layer[i],
                                          64 * (2 ** i))
    else:
        raise Exception("Error : {} is not supported!".format(backbone)) 
    
    output = layers.ConvXD(filters=n_class,
                           kernel_size=1,
                           padding="same",
                           activation=top_activation)(current_layer)

    model = tf.keras.Model(input, output)

    if backbone and gray_input:
        # refer to backbone explanation in vgg16 section above for why I'm setting the weights so late
        if backbone == supported_backbone[0]:
            w, b = backbone_model.layers[1].get_weights()  # rgb weights
            nw = np.expand_dims(w.mean(axis=-2), axis=-2)  # color channels mean, then add gray channel
            model.layers[1].set_weights([nw, b])

    return model