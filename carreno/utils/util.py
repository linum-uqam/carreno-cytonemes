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


def point_distance(coord1, coord2=None, dist=None):
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
    diff = np.array(coord1) - np.array(coord2)
    
    if dist is not None:
        diff = diff * np.array(dist)
    
    return (diff ** 2).sum() ** 0.5
    

def path_length(coordinates, distances=[1, 1, 1]):
    """Get length of a list of coordinates
    Parameters
    ----------
    coordinates : list
        list of coordinates forming a path
    distances : list or ndarray
        axis distances in order
    Returns
    -------
    length : float
        length of path
    """
    length = 0
        
    for j in range(1, len(path)):
        length += utils.point_distance(path[j-1], path[j], distances)
    
    return length