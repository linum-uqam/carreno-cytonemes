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


def UNet(shape, n_class=3, depth=4, n_feat=64, dropout=0.3, norm_order=0,
         activation='relu', top_activation='softmax', backbone=None, pretrained=True):
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
    norm_order : int
        Determines location of batch normalization
        - 0 : batch normalization before activation like in paper
        - 1 : batch normalization after activation like today standards
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
    ndim = len(shape) - 1
    kernel_size_conv =     [3] * ndim
    kernel_size_sampling = [2] * ndim
    gray_input = shape[-1] == 1
    layers = carreno.nn.layers.layers(ndim)
    backbone_model = None

    def one_conv(input, n_feat=32, size=kernel_size_conv, dropout=0.3, activation='relu'):
        """
        1 convolution layers to add with batch normalisation
        Parameters
        ----------
        input : tf.keras.engine.keras_tensor.KerasTensor
            input layer to pick up from
        n_feat : int
            number of features for convolutions
        size : int or list
            conv kernel shape
        dropout : float
            dropout rate
        activation : str
            activation type
        Returns
        -------
        output : tf.keras.engine.keras_tensor.KerasTensor
            last keras layer output
        """
        # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        # See figs_and_nuts answer for layer order
        conv = layers.ConvXD(n_feat,
                             size,
                             padding="same")(input)
        if norm_order:
            # put batch norm after activation
            acti   = layers.Activation(activation)(conv)
            drop   = layers.Dropout(dropout)(acti)
            output = layers.BatchNormalization()(drop)
        else:
            # put batch norm before activation
            norm   = layers.BatchNormalization()(conv)
            acti   = layers.Activation(activation)(norm)
            output = layers.Dropout(dropout)(acti)
        return output

    def two_conv(input, n_feat=32):
        """
        2 convolutions layers to add with batch normalisation
        Parameters
        ----------
        input : tf.keras.engine.keras_tensor.KerasTensor
            input layer to pick up from
        n_feat : int
            number of features for convolutions
        Returns
        -------
        current_layer : tf.keras.engine.keras_tensor.KerasTensor
            last keras layer output
        """
        current_layer = input
        for i in range(2):
            current_layer = one_conv(current_layer,
                                     n_feat=n_feat,
                                     size=kernel_size_conv,
                                     dropout=dropout,
                                     activation=activation)
        return current_layer

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
    
    
    current_layer = layers.ConvXD(filters=n_class,
                           kernel_size=1,
                           padding="same")(current_layer)
    
    output = layers.Activation(top_activation)(current_layer)
    
    if top_activation == 'relu':
        # Based on SoftSeg normalisation of ReLU output
        # https://ivadomed.org/_modules/ivadomed/models.html#Unet
        # Important since ReLU output range goes to infinity n beyond
        normalize = output / tf.reduce_max(output)
        
        # handle division by 0
        output = tf.where(tf.math.is_nan(normalize), tf.zeros_like(normalize), normalize)

        if n_class > 0:
            all_sums = tf.expand_dims(tf.reduce_sum(output, axis=-1), axis=-1)
            all_sums = tf.where(all_sums == 0, tf.ones_like(all_sums), all_sums)
            output = output / all_sums

    model = tf.keras.Model(input, output)

    if backbone and gray_input:
        # refer to backbone explanation in vgg16 section above for why I'm setting the weights so late
        if backbone == supported_backbone[0]:
            w, b = backbone_model.layers[1].get_weights()  # rgb weights
            nw = np.expand_dims(w.mean(axis=-2), axis=-2)  # color channels mean, then add gray channel
            model.layers[1].set_weights([nw, b])

    return model


if __name__ == '__main__':
    import unittest

    class TestUnet(unittest.TestCase):
        def train_n_pred(self, shape, n_class, depth, n_feat, backbone, top_activation='softmax'):
            # generate data
            n = np.prod(np.array(shape[:-1]))
            x = np.arange(n).reshape(shape[:-1])
            classes = x % n_class
            c = [classes == i for i in range(n_class)]
            x = np.stack([x]*shape[-1], axis=-1) / (n-1)  # normalize with division
            y = np.stack([c[0], c[1]], axis=-1) if n_class == 2 else np.stack(c, axis=-1)

            # create architecture
            try:                
                model = UNet(shape=shape,
                             n_class=n_class,
                             depth=depth,
                             n_feat=n_feat,
                             backbone=backbone,
                             pretrained=False,
                             top_activation=top_activation)
            except Exception as e:
                self.fail("Failed creating architecture : {}".format(e))
            
            # compile model
            try:
                model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                              loss=tf.keras.losses.MeanSquaredError())
            except Exception as e:
                self.fail("Model compilation failed : {}".format(e))
            
            # train model
            xs = x[np.newaxis, :]  # otherwise x is iterated over axis 0 in batch
            ys = y[np.newaxis, :]  # ^ same for y
            try:
                history = model.fit(x=xs,
                                    y=ys,
                                    batch_size=1,
                                    epochs=3,
                                    verbose=0)
            except Exception as e:
                self.fail("Training failed : {}".format(e))
            
            # model prediction
            try:
                pred = model.predict(tf.convert_to_tensor(np.stack([x]*3)), verbose=0)
                self.assertGreaterEqual(np.round(tf.reduce_min(pred).numpy(), 5), 0)
                self.assertLessEqual(   np.round(tf.reduce_max(pred).numpy(), 5), 1)
                #self.assertEqual(       np.round(tf.reduce_sum(pred[0]).numpy(), 5), np.prod(ys.shape))
                self.assertEqual(       np.round(tf.reduce_sum(pred, axis=-1).numpy()[tuple([0]*(ys.ndim-1))], 5), 1)
            except Exception as e:
                self.fail("Prediction failed : {}".format(e))
            
            return model

        def test_2D(self):
            # Color channels
            self.train_n_pred(shape=[32,32,1], n_class=3, depth=2, n_feat=8, backbone=None)
            self.train_n_pred(shape=[32,32,3], n_class=3, depth=2, n_feat=8, backbone=None)
            
            # Backbone
            self.train_n_pred(shape=[32,32,1], n_class=3, depth=4, n_feat=64, backbone="vgg16")
            self.train_n_pred(shape=[32,32,3], n_class=3, depth=4, n_feat=64, backbone="vgg16")
            
        def test_3D(self):
            # Color channels
            self.train_n_pred(shape=[32,32,32,1], n_class=3, depth=2, n_feat=8, backbone=None)
            self.train_n_pred(shape=[32,32,32,3], n_class=3, depth=2, n_feat=8, backbone=None)
            
            # Backbone
            self.train_n_pred(shape=[32,32,32,1], n_class=3, depth=4, n_feat=64, backbone="vgg16")
            self.train_n_pred(shape=[32,32,32,3], n_class=3, depth=4, n_feat=64, backbone="vgg16")
        
        def test_encoder_freeze(self):
            model = self.train_n_pred(shape=[32,32,32,1], n_class=3, depth=2, n_feat=8, backbone=None)
            n_trainable = 0
            for layer in model.layers:
                if layer.trainable:
                    n_trainable += 1
            
            encoder_trainable(model, False)
            n_trainable_after = 0
            for layer in model.layers:
                if layer.trainable:
                    n_trainable_after += 1
            self.assertLess(n_trainable_after, n_trainable)

            encoder_trainable(model, True)
            n_trainable_after = 0
            for layer in model.layers:
                if layer.trainable:
                    n_trainable_after += 1
            self.assertEqual(n_trainable_after, n_trainable)
        
        def test_relu_output(self):
            self.train_n_pred(shape=[32,32,1], n_class=3, depth=2, n_feat=8, backbone=None, top_activation='relu')
    
    unittest.main()