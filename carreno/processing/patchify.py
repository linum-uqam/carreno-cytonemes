# -*- coding: utf-8 -*-
import numpy as np
from skimage.transform import resize
from carreno.utils.util import is_2D, is_3D

def patchify(x, patch_shape, mode=2, stride=None):
    """Divide ndarray in patches. x will be resized to best fit patch shape.
    Parameters
    ----------
    x : ndarray
        ndarray to divide
    patch_shape : list
        Patch desired shape
    mode : int
        ndarray resize mode: (default to 2)
            - 0 for `same` padding
            - 1 for nearest neighbors resize
            - 2 for bi-linear resize
            - 3 for bi-cubic resize
    stride : list
        stride between patches, default to patch shape
    Returns
    -------
    patches : list
        Patches for x
    patches_order : list
        Shape of x divided by it's patches
    """
    patch = []
    patch_order = []
    
    if stride is None:
        stride = patch_shape
    
    tw_dim = is_2D(stride)
    resize_shape = []
    
    for i in range(len(patch_shape)):
        nb_patch_over_axis = max(1, (x.shape[i] - patch_shape[i] + stride[i]) / stride[i])
        
        if mode == 0:
            nb_patch_over_axis = int(np.ceil(nb_patch_over_axis))
        else:
            nb_patch_over_axis = int(np.round(nb_patch_over_axis))
        
        patch_order.append(nb_patch_over_axis)
        
        ax_length = nb_patch_over_axis * stride[i] + patch_shape[i] - stride[i]
        resize_shape.append(ax_length)
            
    resized = x.copy()
        
    if mode == 0:
        # add padding to fit
        axis_padding = np.abs(np.array(x.shape) - np.array(resize_shape))
        pad_width = []
        for pad in axis_padding:
            n = int(pad / 2)
            pad_width.append((n, n))
        resized = np.pad(resized, pad_width=pad_width, mode='edge')
    elif mode == 1:
        # nearest neighbors resize
        resized = resize(resized, resize_shape,
                         order=0, preserve_range=True,
                         anti_aliasing=False)
    elif mode == 2:
        # bi-linear
        resized = resize(resized, resize_shape,
                         order=1, preserve_range=False,
                         anti_aliasing=True)
    elif mode == 3:
        # bi-cubic resize
        resized = resize(resized, resize_shape,
                         order=3, preserve_range=False,
                         anti_aliasing=True)
        
    for a, b in zip(range(0, resize_shape[0] - patch_shape[0] + stride[0], stride[0]),
                    range(patch_shape[0], resize_shape[0] + 1, stride[0])):
        for c, d in zip(range(0, resize_shape[1] - patch_shape[1] + stride[1], stride[1]),
                        range(patch_shape[1], resize_shape[1] + 1, stride[1])):
            if tw_dim:
                patch.append(resized[a:b, c:d])
            else:
                for e, f in zip(range(0, resize_shape[2] - patch_shape[2] + stride[2], stride[2]),
                                range(patch_shape[2], resize_shape[2] + 1, stride[2])):
                    patch.append(resized[a:b, c:d, e:f])
    
    return patch, patch_order


def unpatchify(patches, order, stride=None):
    """Reassemble patches into a ndarray.
    Patches overlap choose voxel nearest to patch center
    Parameters
    ----------
    patches : list
        List of patches to reassemble
    order : list
        Patch order for reassembly
    stride : list
        stride between patches, default to patch shape
    Returns
    -------
    nd : ndarray
        Reconstructed ndarray
    """
    patch_shape = patches[0].shape
    patch_center = []
    nd_shape = []
    tw_dim = is_2D(stride)
    
    for i in range(len(order)):
        ax_length = order[i] * stride[i] + patch_shape[i] - stride[i]
        
        center = patch_shape[i] // 2
        patch_center.append([max(0, center - stride[i] // 2),
                             min(patch_shape[i], center + int(np.ceil(stride[i] / 2)))])
        
        nd_shape.append(ax_length)
    
    # add color channels if not grayscale
    if len(order) < len(patches.shape[1:]):
        nd_shape.append(patches.shape[-1])
            
    nd = np.zeros(nd_shape)
    p_i = 0
    
    for x in range(order[0]):
        x_i = x * stride[0]
        x1 = 0 if x == 0 else patch_center[0][0]
        x2 = patch_shape[0] if x == order[0] - 1 else patch_center[0][1]
        for y in range(order[1]):
            y_i = y * stride[1]
            y1 = 0 if y == 0 else patch_center[1][0]
            y2 = patch_shape[1] if y == order[1] - 1 else patch_center[1][1]
            
            if tw_dim:
                nd[x1 + x_i:x2 + x_i,
                   y1 + y_i:y2 + y_i] = patches[p_i][x1:x2, y1:y2]
                
                p_i += 1
            else:
                for z in range(order[2]):
                    z_i = z * stride[2]
                    z1 = 0 if z == 0 else patch_center[2][0]
                    z2 = patch_shape[2] if z == order[2] - 1 else patch_center[2][1]
                    
                    nd[x1 + x_i:x2 + x_i,
                       y1 + y_i:y2 + y_i,
                       z1 + z_i:z2 + z_i] = patches[p_i][x1:x2, y1:y2, z1:z2]
                    
                    p_i += 1
    
    return nd


def volume_pred_from_img(model, x, stride):
    """
    Parameters
    ----------
    model : tf.keras.Model
        model to predict
    x : ndarray
        volume to predict
    stride : [int, int, int]
        stride between patches
    Returns
    -------
    pred_volume : ndarray
        predicted volume
    """
    # get input shape
    patch_shape = [1] + list(model.get_config()["layers"][0]["config"]["batch_input_shape"][1:-1])
    
    # patchify to fit inside model input
    patch, order = patchify(x, patch_shape=patch_shape, mode=2, stride=stride)
    
    # remove z axis (we want an image) and add color channel
    preprocess = np.expand_dims(np.squeeze(np.array(patch), axis=1), axis=-1)

    # predict patch segmentation
    pred_patch = model.predict(preprocess)

    # add z axis again
    postprocess = np.expand_dims(pred_patch, axis=1)

    # reassemble patches into a volume
    pred_volume = unpatchify(postprocess, order=order, stride=stride)

    return pred_volume


def volume_pred_from_vol(model, x, stride):
    """
    Parameters
    ----------
    model : tf.keras.Model
        model to predict
    x : ndarray
        volume to predict
    stride : [int, int, int]
        stride between patches
    Returns
    -------
    pred_volume : ndarray
        predicted volume
    """
    # get input shape
    patch_shape = list(model.get_config()["layers"][0]["config"]["batch_input_shape"][1:-1])

    # patchify to fit inside model input
    patch, order = patchify(x, patch_shape=patch_shape, mode=2, stride=stride)
    
    # add color channel
    preprocess = np.expand_dims(np.array(patch), axis=-1)

    # predict patch segmentation
    pred_patch = model.predict(preprocess)

    # reassemble patches into a volume
    pred_volume = unpatchify(pred_patch, order=order, stride=stride)

    return pred_volume