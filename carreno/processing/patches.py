# -*- coding: utf-8 -*-
import numpy as np
from skimage.transform import resize
import patchify as p
import scipy

from carreno.utils.util import is_2D, is_3D

def neighbors_steps(x, step, stride=None):
    """
    Get neighboring step for X to know how many Y can fit with strides of Z
    Parameters
    ----------
    x : float
        number
    step : float
        step size
    stride : float
        stride
    Returns
    -------
    neighbors : [int, int, int]
        previous step, next step and closed step
    """
    if stride is None:
        stride = step

    n_steps = 1  # nb of steps, need at least 1 to avoid empty axis
    if x > step:
        # find how many more patches we can fit in X
        n_steps = ((x - step) / stride) + 1

    # we don't want negative patches if step size is too big
    prev_step = np.floor(n_steps)
    next_step = np.ceil(n_steps)
    neighbors = [(i + step) * stride for i in [prev_step, next_step]]
    closest_step = n_steps - prev_step >=0.5

    return neighbors + [closest_step]


def pad_n_slice_img(x, step, stride=None, position=2, mode="constant", constant_values=0):
    """
    Pad or slice X to match step size. Pad if axis is not long enough, slice if too long
    Parameters
    x : array-like
        Array to adjust shape
    step : [int]
        Step size per axis (equivalent to patches axis length)
    stride : [int]
        Patches stride per axis
    position : int
        Where the padding or slicing will occur
         - 0 : at the beginning of axis
         - 1 : in the middle of axis
         - 2 : at the end of axis
    mode : str
        Refer to `mode` param from numpy.pad method
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    constant_values : float
        Refer to `constant_values` param from numpy.pad method
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    Returns
    -------
    y : array-like
        Padded and sliced x input
    """
    y = np.array(x)
    shape = x.shape[:len(step)]
    pad_width = [[0,0]] * len(y.shape)  # default padding
    #slices = [slice(start, end) for start, end in zip([0] * len(step), shape)]  # default slicing

    for ax in range(len(shape)):
        n = neighbors_steps(shape[ax], step=step[ax], stride=stride[ax])
        recommended_size = min(step, n[2])  # we want at least 1 patch
        dif = ax - recommended_size

        if dif < 0:
            dif = abs(dif)

            if position == 0:
                pad_width[ax] = [dif, 0]
            elif position == 1:
                half = dif//2
                pad_width[ax] = [half, half + (dif & 1)]  # add extra 1 if odd
            else:
                pad_width[ax] = [0, dif]
        
        elif dif > 0:
            slicing = None
            
            if position == 0:
                slicing = slice(dif, shape[ax])
            elif position == 1:
                half = dif//2
                slicing = slice(half, half + (dif & 1))
            else:
                slicing = slice(0, -dif)
            
            # we can already start slicing for efficiency
            y = np.delete(y, slicing, ax)

    y = np.pad(y, pad_width=pad_width, mode=mode, constant_values=constant_values)
    
    return y
    

def patchify(x, patch_shape, stride=None, resize_mode=2, pad_mode='constant'):
    """Divide ndarray in patches. x will be resized to best fit patch shape.
    Parameters
    ----------
    x : ndarray
        ndarray to divide
    patch_shape : list
        Patch desired shape
    stride : list
        stride between patches, default to patch shape
    resize_mode : int
        ndarray resize mode: (default to 2)
            - 0 for padding
            - 1 for nearest neighbors resize
            - 2 for bi-linear resize
            - 3 for bi-cubic resize
    pad_mode : str
        padding mode, see numpy.pad for more info
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    Returns
    -------
    patches : list
        Patches for x with order shape
    """
    if stride is None:
        stride = patch_shape
    
    resize_shape = []
    
    for i in range(len(patch_shape)):
        nb_patch_over_axis = max(1, (x.shape[i] - patch_shape[i] + stride[i]) / stride[i])
        
        if resize_mode == 0:
            nb_patch_over_axis = int(np.ceil(nb_patch_over_axis))
        else:
            nb_patch_over_axis = int(np.round(nb_patch_over_axis))
        
        ax_length = nb_patch_over_axis * stride[i] + patch_shape[i] - stride[i]
        resize_shape.append(ax_length)
            
    resized = x.copy()
        
    if resize_mode == 0:
        # add padding to fit
        axis_padding = np.abs(np.array(x.shape) - np.array(resize_shape))
        pad_width = []
        for pad in axis_padding:
            n = pad // 2
            pad_width.append((n, n + (pad & 1)))
        resized = np.pad(resized, pad_width=pad_width, mode=pad_mode)
    elif resize_mode == 1:
        # nearest neighbors resize
        resized = resize(resized, resize_shape,
                         order=0, preserve_range=True,
                         anti_aliasing=False)
    elif resize_mode == 2:
        # bi-linear
        resized = resize(resized, resize_shape,
                         order=1, preserve_range=False,
                         anti_aliasing=True)
    elif resize_mode == 3:
        # bi-cubic resize
        resized = resize(resized, resize_shape,
                         order=3, preserve_range=False,
                         anti_aliasing=True)

    print(resized.shape)

    return p.patchify(resized, patch_size=patch_shape, step=stride)


def unpatchify_w_nearest(patches, order, stride=None):
    """
    Reassemble patches into a ndarray.
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
    pred_volume = unpatchify_w_nearest(postprocess, order=order, stride=stride)

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
    pred_volume = unpatchify_w_nearest(pred_patch, order=order, stride=stride)

    return pred_volume


def unpatchify_w_gaussian(patches, order, patch_shape, stride=None, weight=None):
    """
    Reassemble patches, with intensity range between 0 and 1, into a ndarray.
    Use a gaussian distribution to avg patches overlap. Everything is softmax at the end.
    Parameters
    ----------
    patches : list
        Iterator for of patches to reassemble
        (avoid loading every patches at once for the sake of your memory)
    order : list
        Patch order for reassembly
    stride : list
        stride between patches, default to patch shape
    weight : array-like
        weight for each value in patches
    Returns
    -------
    nd : ndarray
        Reconstructed ndarray
    """
    nd_shape = []
    tw_dim = is_2D(stride)

    for i in range(len(order)):
        ax_length = patch_shape[i] + (order[i] - 1) * stride[i]
        nd_shape.append(ax_length)
    
    # add color channels if not grayscale
    if len(order) < len(patch_shape):
        nd_shape + list(patch_shape[len(order):])
    
    nd = np.zeros(nd_shape, dtype=float)
    p_i = 0
    
    for x in range(nd_shape[0]):
        x1 = x * stride[0]
        x2 = x1 + patch_shape[0]
        for y in range(nd_shape[1]):
            y1 = y * stride[0]
            y2 = y1 + patch_shape[0]
            
            if tw_dim:
                nd[x1:x2,
                   y1:y2] = next(patches) * weight
                
                p_i += 1
            else:
                for z in range(nd_shape[2]):
                    z1 = z * stride[0]
                    z2 = z1 + patch_shape[0]
                    
                    nd[x1:x2,
                       y1:y2,
                       z1:z2] = next(patches) * weight
                    
                    p_i += 1

    # normalize over 1
    nd_min = nd.min()
    nd_max = nd.max() - nd_min
    nd = (nd - nd_min) / nd_max

    nd = scipy.special.softmax(nd, axis=-1)

    # normalize back to original intensities
    nd = nd * nd_max + nd_min

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

    # gaussian distribution
    weight = ...

    # reassemble patches into a volume
    pred_volume = unpatchify_w_gaussian(postprocess,
                                        patch_shape=patch_shape,
                                        order=order,
                                        stride=stride,
                                        weight=weight)

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

    # gaussian distribution
    weight = ...

    # reassemble patches into a volume
    pred_volume = unpatchify_w_gaussian(pred_patch,
                                        patch_shape=patch_shape,
                                        order=order,
                                        stride=stride,
                                        weight=weight)

    return pred_volume