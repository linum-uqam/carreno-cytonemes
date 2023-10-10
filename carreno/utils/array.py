# -*- coding: utf-8 -*-
import numpy as np

def normalize(x, minv=0, maxv=1):
    """
    Normalize array between given range
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


def standardize(x):
    """
    Standardize array
    Parameters
    ----------
    x : list ndarray
    Returns
    -------
    y : ndarray
        standardize x with dtype float32
    """
    mean = x.mean()
    std = x.std()
    y = x - mean / std
    return y


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


def butterworth(n, low, high, shift=0):
    """
    Butterworth distribution in 1D
    Parameters
    ----------
    n : int
        length of the distribution
    low : float
        low values boundaries
    high : float
        high values boundaries
    shift : int
        shift distribution right (+) or left (-)
    Returns
    -------
    n : ndarray
        distribution values in a list
    """
    u = np.linspace(0-shift, n-shift, n)
    D = np.sqrt((u-n/2)**2)
    h = 1 / (1 + (D/low)**(2*high))
    return h