import numpy as np


def point_distance(coord1, coord2=None, dist=None):
    """Distance between 2 points
    Get the Euclidean distance between 2 points
    
    :param coord1: coordonate of first point  (list)
    :param coord2: coordonate of second point, default is the absolute 0 coord (list)
    :param dist:   length between each pixel, default is 1 for each axes (list)
    :return:       distance between coord1 and coord2 
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