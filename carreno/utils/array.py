# -*- coding: utf-8 -*-
import numpy as np

def normalize(x, minv=0, maxv=1):
    """Normalize array between given range
    Parameters
    ----------
    x : list, ndarray
        array to normalize
    minv : float
        min value
    maxv : float
        max value
    Returns
    -------
    y : ndarray
        normalized x with dtype float32
    """
    y = np.array(x, dtype=np.float32)
    y = y + (0 - y.min())
    y = y / y.max()
    return y * (maxv - minv) + minv


def euclidean_dist(coord1, coord2=0, dist=None):
    """Get the Euclidean distance between 2 points
    Parameters
    ----------
    coord1 : list
        coordonate of first point
    coord2 : list
        coordonate of second point, default is the absolute 0 coord
    dist : list
        length between each pixel, default is 1 for each axes
    Returns
    -------
    __ : float
        distance between coord1 and coord2 
    """
    diff = np.array(coord1) - coord2
    
    if dist is not None:
        diff = diff * dist
    
    return (diff ** 2).sum() ** 0.5
    

def unstack(a, axis=0):
    """
    Unstack ndarray. The Opposite of numpy.stack()

    :param a: array like ndarray
    :param axis: axis to unstack
    :return: unstacked array
    """
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis=axis)]
    

def coord2np(coord):
    """coordinate to numpy index
    [z, y, x] -> [[z], [y], [x]]
    
    :param coord: coordinate (list)
    :return:      numpy index (list)
    """
    return tuple([[i] for i in coord])


def nb_color_channel(shape):
    """
    Assuming shape is for an standard image or a volume
    Parameters
    ----------
    shape : [int]
        Shape of array-like
    Returns
    -------
    _ : int
        Number of color channels, 0 if grayscale
    """
    if len(shape) == 2 or (len(shape) == 3 and shape[-1] > 4):
        return 0
    return shape[-1]


def ndim_for_pixel(shape):
    """
    Get number of axis in shape without color channels
    Parameters
    ----------
    shape : [int]
        Shape of array-like
    Returns
    -------
    _ : bool
        if shape could be for 2D image
    """
    # check if grayscale, rgb or rgba
    nch = nb_color_channel(shape)
    return len(shape) - min([nch, 1])  # either minus 0 or 1

# TODO generate a gaussian distribution?
#def gaussian_filter(sigma=1.0, mu=0.0):
#    # https://stackoverflow.com/questions/14873203/plotting-of-1-dimensional-gaussian-distribution-function
#
#    from matplotlib import pyplot as mp
#    import numpy as np
#
#    def gaussian(x, mu, sig):
#        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#
#    x_values = np.linspace(-3, 3, 120)
#    for mu, sig in [(-1, 1), (0, 2), (2, 3)]:
#        mp.plot(x_values, gaussian(x_values, mu, sig))
#
#    mp.show()
#    return ...