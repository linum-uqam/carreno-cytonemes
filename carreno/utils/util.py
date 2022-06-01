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
        normalized x
    """
    y = np.array(x)
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
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)]
    

def coord2np(coord):
    """coordinate to numpy index
    [z, y, x] -> [[z], [y], [x]]
    
    :param coord: coordinate (list)
    :return:      numpy index (list)
    """
    return tuple([[i] for i in coord])