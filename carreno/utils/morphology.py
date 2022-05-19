import numpy as np
from scipy import ndimage as nd

def getNeighbors3D(index, x):
    """Get neighbors around the index in data with a connectivity of 26.
    Less flexible, but faster than the implementation in utils
    Parameters
    ----------
    index : list
        coordinate axes position
    x : ndarray
        binary volume
    structure : list
        connectivity structure, default is full
    Returns
    -------
    neighbors : ndarray
        neighbors coordinates where each instance is a coordinate (same format as index param)
    """
    shape = x.shape

    # slicing param
    slice = []
    center = [1, 1, 1]
    for i in range(3):
        start = index[i] - 1
        if 0 > start:
            start = 0
            center[i] = 0

        end = min(shape[i], index[i] + 1)
        slice.append((start, end))

    # get neighbors zone in a copy of x
    neighbors_zone = np.array(x[slice[0][0]:slice[0][1] + 1,
                                slice[1][0]:slice[1][1] + 1,
                                slice[2][0]:slice[2][1] + 1])

    # no self connectivity
    neighbors_zone[tuple(center)] = 0

    neighbors = np.stack(np.where(neighbors_zone >= 1), axis=1)

    # re-center coord
    neighbors = neighbors - center

    # coord from index position
    return neighbors + np.array(index)


def getNeighbors(index, data, structure=None):
    """Get neighbors for index
    Get neighbors around the index in data
    
    :param index:        coordonate of point (list)
    :param data:         binary image or volume (ndarray)
    :param structure:    connectivity structure, default is full (list)
    :return:             neighbors coordinates (ndarray)
    """
    index_c = []
    for axis_i in index:
        index_c.append([int(axis_i)])
    index_c = tuple(index_c)

    struct = None
    if structure is None:
        nb_axis = len(data.shape)
        struct = np.ones([3] * nb_axis)
    struct = np.array(structure)

    neighbors_zone = np.zeros_like(data)
    neighbors_zone[index_c] = 1
    neighbors_zone = nd.morphology.binary_dilation(neighbors_zone,
                                                   structure=struct,
                                                   mask=data)
    neighbors_zone[index_c] = 0
    
    return np.stack(np.where(neighbors_zone == 1), axis=1)


def position_inside(position, volume):
    """Check if position is inside object in volume
    Parameters
    ----------
    volume : ndarray
        binary 3d volume
    position : list
        position to validate inside the volume
    Returns
    -------
    __ : bool
        if valid or not
    """
    pos = np.round(np.array(position)).astype(int)
    maximum = np.array(volume.shape)
    
    # out of bound?
    if (pos < np.array([0,0,0])).any() or (pos >= maximum).any():
        return False
    
    # inside volume object?
    return volume[pos[0], pos[1], pos[2]] == True