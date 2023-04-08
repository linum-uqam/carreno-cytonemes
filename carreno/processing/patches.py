# -*- coding: utf-8 -*-
import numpy as np
from skimage.transform import resize
import patchify as p

from carreno.utils.array import ndim_for_pixel


def reshape_patchify(x, n_axis=3):
    """
    Instead of having patch order inside X shape, put patches over
    the same axis for easy iteration and save order.
    Parameters
    ----------
    x : array-like
        Patches with patchify output format
    n_axis : int
        Number of axis in patch shape
    Returns
    -------
    y : array-like
        Patches with flatten axis for iteration
    order : tuple
        Patches order for unpatchify later
    """
    order = x.shape[:-n_axis]
    y = x.reshape([np.prod(order)] + list(x.shape[-n_axis:]))
    return y, order


def patchify(x, patch_shape, stride=None, resize_mode=2, pad_mode='constant', constant_values=0):
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
        xax = x.shape[i]
        pax = patch_shape[i]
        sax = stride[i]
        nb_patch_over_axis = max(1, (xax - pax + sax) / sax)
        
        if resize_mode == 0:
            nb_patch_over_axis = int(np.ceil(nb_patch_over_axis))
        else:
            nb_patch_over_axis = int(np.round(nb_patch_over_axis))
        
        ax_length = nb_patch_over_axis * sax + pax - sax
        resize_shape.append(ax_length)

    resized = x.copy()
        
    if resize_mode == 0:
        # add padding to fit
        axis_padding = np.abs(np.array(x.shape) - np.array(resize_shape))
        pad_width = []
        for pad in axis_padding:
            n = pad // 2
            pad_width.append((n, n + (pad & 1)))
        resized = np.pad(resized, pad_width=pad_width, mode=pad_mode, constant_values=constant_values)
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

    return p.patchify(resized, patch_size=patch_shape, step=stride)


def unpatchify_w_nearest(patches, order, patch_shape, stride=None):
    """
    Reassemble patches into a ndarray.
    Patches overlap choose voxel nearest to patch center
    Parameters
    ----------
    patches : list
        Iterator for of patches to reassemble
        (avoid loading every patches at once for the sake of your memory)
    order : list
        Patch order for reassembly
    patch_shape : list
        Patch shape
    stride : list
        stride between patches, default to patch shape
    Returns
    -------
    nd : ndarray
        Reconstructed ndarray
    """
    patch_shape = patch_shape
    patch_center = []
    nd_shape = []
    tw_dim = ndim_for_pixel(patch_shape) == 2
    
    for i in range(len(order)):
        ax_length = order[i] * stride[i] + patch_shape[i] - stride[i]
        
        center = patch_shape[i] // 2
        patch_center.append([max(0, center - stride[i] // 2),
                             min(patch_shape[i], center + int(np.ceil(stride[i] / 2)))])
        
        nd_shape.append(ax_length)
    
    # add color channels if not grayscale
    if len(order) < len(patch_shape):
        nd_shape = nd_shape + patch_shape[len(order):]
            
    nd = np.zeros(nd_shape)
    
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
                   y1 + y_i:y2 + y_i] = next(patches)[x1:x2, y1:y2]
            else:
                for z in range(order[2]):
                    z_i = z * stride[2]
                    z1 = 0 if z == 0 else patch_center[2][0]
                    z2 = patch_shape[2] if z == order[2] - 1 else patch_center[2][1]
                    
                    nd[x1 + x_i:x2 + x_i,
                       y1 + y_i:y2 + y_i,
                       z1 + z_i:z2 + z_i] = next(patches)[x1:x2, y1:y2, z1:z2]
    
    return nd


def unpatchify_w_weight(patches, order, patch_shape, stride=None, weight=None):
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
    patch_shape : list
        Patch shape
    stride : list
        stride between patches, default to patch shape
    weight : array-like
        weight for each value in patches
    Returns
    -------
    nd : ndarray
        Reconstructed float ndarray between 0 and 1
    """
    if weight is None:
        weight = np.ones(patch_shape)
    
    nd_shape = []
    tw_dim = ndim_for_pixel(stride) == 2

    for i in range(len(order)):
        ax_length = patch_shape[i] + (order[i] - 1) * stride[i]
        nd_shape.append(ax_length)
        
    # add color channels if not grayscale
    if len(order) < len(patch_shape):
        nd_shape = nd_shape + patch_shape[len(order):]
    
    nd = np.zeros(nd_shape, dtype=float)
    nd_weight = nd.copy()
    
    for x in range(order[0]):
        x1 = x * stride[0]
        x2 = x1 + patch_shape[0]
        for y in range(order[1]):
            y1 = y * stride[1]
            y2 = y1 + patch_shape[1]

            if tw_dim:
                weighted_patch = next(patches) * weight
                nd[x1:x2,
                   y1:y2] += weighted_patch
                nd_weight[x1:x2,
                          y1:y2] += weight
            else:
                for z in range(order[2]):
                    z1 = z * stride[2]
                    z2 = z1 + patch_shape[2]
                    
                    weighted_patch = next(patches) * weight
                    nd[x1:x2,
                       y1:y2,
                       z1:z2] += weighted_patch
                    nd_weight[x1:x2,
                              y1:y2,
                              z1:z2] += weight

    # softmax instead of normalize sounded like a good idea at first for
    # unpatchify on predictions, but the results are awful with normal
    # inputs

    return nd / nd_weight


def __pred_iterator(model, xs, is_img=True):
    """
    Iterator for model prediction
    Parameters
    ----------
    model : tf.keras.Model
        Prediction model
    xs : list
        input to predict
    is_img : bool
        are the inputs images or volumes?
    Returns
    -------
    pred_patch : array-like
        model prediction for current output in iterator
    """
    for x in xs:
        # predict patch segmentation one at a time
        pred_patch = model.predict(np.array([x]))

        # rm batch axis
        if not is_img:
            # for img, batch axis will be interpreted as z axis of 1
            pred_patch = np.squeeze(pred_patch, axis=0)
        
        yield pred_patch


def volume_pred_from_img(model, x, stride, weight=None):
    """
    Parameters
    ----------
    model : tf.keras.Model
        model to predict
    x : ndarray
        volume to predict
    stride : [int]
        stride between patches without color axis
    Returns
    -------
    pred_volume : ndarray
        predicted volume
    """
    # get input shape
    #patch_shape = [1] + list(model.get_config()["layers"][0]["config"]["batch_input_shape"][1:-1])  # weird flex but okay
    input_patch_shape = [1] + list(model.layers[0].input.shape[1:])
    input_stride = list(stride) + input_patch_shape[len(stride):]
    
    # patchify to fit inside model input
    patch = patchify(x,
                     patch_shape=input_patch_shape,
                     stride=input_stride,
                     resize_mode=0)
    patch, order = reshape_patchify(patch, len(input_patch_shape))
    
    # remove z axis (we want an image)
    preprocess = np.squeeze(np.array(patch), axis=1)
    
    # to avoid memory issues, use an iterator for prediction
    iterator = __pred_iterator(model, preprocess, True)

    # get output shape
    output_patch_shape = [1] + list(model.layers[-1].output.shape[1:])
    output_stride = list(stride) + [output_patch_shape[-1]]
    
    # reassemble patches into a volume
    pred_volume = None
    if weight is None:
        pred_volume = unpatchify_w_nearest(iterator,
                                           patch_shape=output_patch_shape,
                                           order=order,
                                           stride=output_stride)
    else:
        pred_volume = unpatchify_w_weight(iterator,
                                          patch_shape=output_patch_shape,
                                          order=order,
                                          stride=output_stride,
                                          weight=weight)

    return pred_volume


def volume_pred_from_vol(model, x, stride, weight=None):
    """
    Parameters
    ----------
    model : tf.keras.Model
        model to predict
    x : ndarray
        volume to predict
    stride : [int]
        stride between patches without color axis
    Returns
    -------
    pred_volume : ndarray
        predicted volume
    """
    # get input shape
    input_patch_shape = list(model.layers[0].input.shape[1:])
    input_stride = list(stride) + input_patch_shape[len(stride):]

    # patchify to fit inside model input
    patch = patchify(x,
                     patch_shape=input_patch_shape,
                     stride=input_stride,
                     resize_mode=0)
    patch, order = reshape_patchify(patch, len(input_patch_shape))
    
    # to avoid memory issues, use an iterator for prediction
    iterator = __pred_iterator(model, patch, False)
    
    # get output shape
    output_patch_shape = model.layers[-1].output.shape[1:]
    output_stride = list(stride) + output_patch_shape[len(stride):]

    # reassemble patches into a volume
    pred_volume = None
    if weight is None:
        pred_volume = unpatchify_w_nearest(iterator,
                                           patch_shape=output_patch_shape,
                                           order=order,
                                           stride=output_stride)
    else:
        pred_volume = unpatchify_w_weight(iterator,
                                          patch_shape=output_patch_shape,
                                          order=order,
                                          stride=output_stride,
                                          weight=weight)
    
    return pred_volume