# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from carreno.nn.unet import UNet


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
                weights3D[0][:] = weight2D_to_3D(weights2D, weights2D[0].shape[0])
                
                layer3D.set_weights(weights3D)
            except:
                # probably a pooling operation without weights
                #print('Could not transfer layer', name2D, 'to', name3D)
                pass
    
    return unet3D