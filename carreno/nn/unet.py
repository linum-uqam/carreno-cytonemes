# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import carreno.nn.layers


def __add_attr_n_meth(model, depth, backbone):
    """
    Add custom attributes and methods to my UNet model
    Parameters
    ----------
    model : tf.keras.Model
        UNet model to update
    depth : int
        UNet depth
    backbone : str, None
        UNet backbone
    Returns
    -------
    None
    """
    
    """
    Extra attributes for Keras model
    Attributes
    ----------
    depth : int
        UNet depth
    backbone : str, None
        Backbone model type
    encoder_last_layer : int
        How many layer until decoder (includes middle block)
    """
    
    model.depth = depth
    model.backbone = backbone
    
    # how many layer until decoder (includes middle block)
    if backbone == "vgg16":
        model.encoder_last_layer = 17
    else:
        # assume it's our default UNet architecture
        model.encoder_last_layer = depth * 7 - 1
    
    """
    Extra methods for Keras model
    Methods
    -------
    train_encoder :
        Freeze or unfreeze layers before `encoder_last_layer`
    add_backbone :
        Switch out every layer before `encoder_last_layer` for selected backbone
    """
    
    def train_unet_encoder_meth(model, trainable):
        """
        Freeze or unfreeze layers before `encoder_last_layer`.
        Parameters
        ----------
        model : tf.Keras.Model
            UNet model made with my custom param and architecture
        trainable : bool
            True to make encoder trainable, False for untrainable
        """
        carreno.nn.layers.train_encoder(model, start=0, end=model.encoder_last_layer, trainable=trainable)
        return
    
    model.train_encoder = train_unet_encoder_meth.__get__(model)
    model.add_backbone = add_backbone.__get__(model)

    return


def UNet(shape, n_class=3, depth=3, n_feat=32):
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
    depth = depth
    ndim = len(shape) - 1
    kernel_size_conv =     [3] * ndim
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
        skip_concatenate = layers.Concatenate()([skip, upsample])
        out = two_conv(skip_concatenate, n_feat)
        return out
    
    input = layers.Input(shape)
    skip_layer = []
    current_layer = input

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

    __add_attr_n_meth(model, depth=depth, backbone=None)

    return model


def add_backbone(model, backbone, pretrained=True):
    """
    Add a backbone to the UNet model.
    Switch out every layer before `encoder_last_layer` for selected backbone.
    Parameters
    ----------
    model : tf.keras.Model
        UNet model made with my custom param and architecture
    backbone : str
        Backbone type (only vgg16 supported for now)
         - "vgg16" : VGG16 encoder
    pretrained : bool
        Is the backbone be pretrained on imagenet.
        If it's and the decoder isn't, it's recommended to freeze the encoder
        for the first few epoch.
    Returns
    -------
    backbone_model : tf.keras.Model
        New UNet model
    """
    supported_backbone = ['vgg16']

    if not backbone in supported_backbone:
        raise Exception("Error : Backbone " + str(backbone) + " isn't supported.")

    input_shape = list(model.layers[0].input.shape[1:])
    dim = 2 if len(input_shape) == 3 else 3
    gray_input = input_shape[-1] == 1
    encoder_end = model.encoder_last_layer + 1
    
    current_layer = model.layers[0].output
    skip_layers = []
    backbone_model = None

    if backbone == supported_backbone[0]:
        if model.depth != 5:
            raise Exception("Error : UNet architecture incompatible, depth must be 5.")

        backbone_model = tf.keras.applications.VGG16(include_top=False,
                                                     weights='imagenet' if pretrained else None,
                                                     input_shape=input_shape[-3:-1]+[3])
        # rm last layer which is a max pooling
        backbone_model = tf.keras.Model(backbone_model.layers[0].input,
                                        backbone_model.layers[-2].output)
        
        if dim == 3:
            # 2D vgg16 to 3D
            backbone_model = carreno.nn.layers.model2D_to_3D(backbone_model, 3)

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
            current_layer = tf.keras.layers.Conv2D(64, 3,
                                                   padding="same")(current_layer)
        else:
            # use rgb weights and bias
            current_layer = backbone_model.layers[1](current_layer)
        
        for i in range(2, len(backbone_model.layers)-1):  # we skip last layer since it's pooling with no weights
            if "pool" in backbone_model.layers[i].name:
                # skip layer should be the last convolution layer before pooling
                skip_layers.append(current_layer)
            current_layer = backbone_model.layers[i](current_layer)
        
        # update last encoder layer index
        model.encoder_last_layer = len(backbone_model.layers) - 1

    # decoder
    skip_layers_i = -1
    for i in range(encoder_end, len(model.layers)):
        if isinstance(model.layers[i], tf.keras.layers.Concatenate):
            # get parent index in 2D, they should still match in 3D
            combination = [skip_layers[skip_layers_i], current_layer]
            skip_layers_i -= 1
            current_layer = tf.keras.layers.Concatenate()(combination)
        else:
            current_layer = model.layers[i](current_layer)  # THIS WILL NEVER WORK BECAUSE THE CONV_TRANSPOSE GOES FROM 1024 IN UNET TO 512 IN VGG16 FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

    unet_w_backbone = tf.keras.Model(model.layers[0].input, current_layer)

    if backbone and gray_input:
        # refer to backbone explanation in vgg16 section above for why I'm setting the weights so late
        if backbone == supported_backbone[0]:
            w, b = backbone_model.layers[1].get_weights()
            nw = np.expand_dims(w.mean(axis=-1), axis=-1)  # color channels avg, then add gray channel
            unet_w_backbone.layers[1].set_weights([nw, b])
    
    __add_attr_n_meth(unet_w_backbone, model.depth, backbone)
    
    return unet_w_backbone