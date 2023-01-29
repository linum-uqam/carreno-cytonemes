# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from argparse import Namespace


def layers(ndim):
    """
    Namespace for tf.keras.layers used to abstract the need to specify the layer dimension
    Parameters
    ----------
    ndim : int
        number of dimension
    Returns
    -------
    layers : Layers
        namespace for tf.keras.layers with added XD functions which abstract dimension
    """
    # https://stackoverflow.com/questions/39376763/in-python-how-to-deep-copy-the-namespace-obj-args-from-argparse
    # to avoid passing tf.keras.layers as a reference
    layers = Namespace(**vars(tf.keras.layers))

    # add XD functions lol
    if ndim == 2:
        layers.ConvXD =          tf.keras.layers.Conv2D
        layers.MaxPoolingXD =    tf.keras.layers.MaxPooling2D
        layers.ConvXDTranspose = tf.keras.layers.Conv2DTranspose
    elif ndim == 3:
        layers.ConvXD =          tf.keras.layers.Conv3D
        layers.MaxPoolingXD =    tf.keras.layers.MaxPooling3D
        layers.ConvXDTranspose = tf.keras.layers.Conv3DTranspose

    return layers


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
    parent_i = []
    layer_names = [l.name for l in model.layers]
    for name in parent_names:
        try:
            parent_i.append(layer_names.index(name))
        except:
            pass
    
    return parent_i


def model2D_to_3D(model, inp_ndim=64):
    """
    Convert a model 2D layers to 3D while conserving weights and bias.
    We assume the new axis for convolutions kernel is uniform to other dimension.
    Evidently, this can't work with most models, this is meant specifically for an UNet.
    Parameters
    ----------
    model : tf.keras.Model
        keras 2D model to convert
    inp_ndim : int
        input new axis length
    Returns
    -------
    nmodel : tf.keras.Model
        keras 3D model
    """
    nshape = [inp_ndim, *model.layers[0].input_shape[0][1:]]
    layers2D = layers(2)
    layers3D = layers(3)

    # x contains all the layers
    x = [layers3D.Input(nshape)]

    # process hidden layers
    conv_weights_transfer = []
    batch_norm_weights_transfer = []
    for i in range(1, len(model.layers)):
        l = model.layers[i]
        if isinstance(l, layers2D.ConvXDTranspose):  # need to check if instance is ConvXDTranspose before ConvXD
            conv_weights_transfer.append(i)
            conv_trans_layer = layers3D.ConvXDTranspose(filters=l.filters,
                                                        kernel_size=[l.kernel_size[0], *l.kernel_size],  # add uniform new dim
                                                        strides=[l.strides[0], *l.strides],
                                                        padding=l.padding,
                                                        output_padding=l.output_padding,
                                                        data_format=l.data_format,
                                                        dilation_rate=[l.dilation_rate[0], *l.dilation_rate],
                                                        groups=l.groups,
                                                        activation=l.activation,
                                                        use_bias=l.use_bias,
                                                        kernel_initializer=l.kernel_initializer,
                                                        bias_initializer=l.bias_initializer,
                                                        kernel_regularizer=l.kernel_regularizer,
                                                        bias_regularizer=l.bias_regularizer,
                                                        activity_regularizer=l.activity_regularizer,
                                                        kernel_constraint=l.kernel_constraint,
                                                        bias_constraint=l.bias_constraint)
            x.append(conv_trans_layer(x[-1]))
        elif isinstance(l, layers2D.ConvXD):
            conv_weights_transfer.append(i)
            conv_layer = layers3D.ConvXD(filters=l.filters,
                                         kernel_size=[l.kernel_size[0], *l.kernel_size],  # add uniform new dim
                                         strides=[l.strides[0], *l.strides],
                                         padding=l.padding,
                                         data_format=l.data_format,
                                         dilation_rate=[l.dilation_rate[0], *l.dilation_rate],
                                         groups=l.groups,
                                         activation=l.activation,
                                         use_bias=l.use_bias,
                                         kernel_initializer=l.kernel_initializer,
                                         bias_initializer=l.bias_initializer,
                                         kernel_regularizer=l.kernel_regularizer,
                                         bias_regularizer=l.bias_regularizer,
                                         activity_regularizer=l.activity_regularizer,
                                         kernel_constraint=l.kernel_constraint,
                                         bias_constraint=l.bias_constraint)
            x.append(conv_layer(x[-1]))
        elif isinstance(l, layers2D.MaxPoolingXD):
            max_pool_layer = layers3D.MaxPoolingXD(pool_size=[l.pool_size[0], *l.pool_size],
                                                   strides=[l.strides[0], *l.strides],
                                                   padding=l.padding,
                                                   data_format=l.data_format)
            x.append(max_pool_layer(x[-1]))
        elif isinstance(l, layers2D.Concatenate):
            # get parent index in 2D, they should still match in 3D
            parents2D_idx = get_layer_parent_i(l, model)
            parents3D_layers = [x[j] for j in parents2D_idx]
            x.append(layers3D.Concatenate()(parents3D_layers))
        elif isinstance(l, layers2D.BatchNormalization):
            batch_norm_weights_transfer.append(i)
            batch_norm = layers3D.BatchNormalization(axis=l.axis[0]+1,  # adding an axis for shape, usually the last one
                                                     momentum=l.momentum,
                                                     epsilon=l.epsilon,
                                                     center=l.center,
                                                     scale=l.scale,
                                                     beta_initializer=l.beta_initializer,
                                                     moving_variance_initializer=l.moving_variance_initializer,
                                                     beta_regularizer=l.beta_regularizer,
                                                     gamma_regularizer=l.gamma_regularizer,
                                                     beta_constraint=l.beta_constraint,
                                                     gamma_constraint=l.gamma_constraint)
            x.append(batch_norm(x[-1]))
        else:
            # hope for the best and use layer as it is
            x.append(l(x[-1]))

    nmodel = tf.keras.Model(x[0], x[-1])

    # transfer learning for conv
    for i in conv_weights_transfer:
        w2D, bias = model.layers[i].get_weights()
        ndim = nmodel.layers[i].kernel_size[0]
        w3D = weight2D_to_3D(w2D, ndim)
        nmodel.layers[i].set_weights([w3D, bias])

    # transfer learning for batch norm
    for i in batch_norm_weights_transfer:
        nmodel.layers[i].set_weights(model.layers[i].get_weights())

    return nmodel