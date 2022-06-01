import numpy as np
from scipy import ndimage as nd

def getNeighbors(index, data, structure=None):
    """Get neighbors around the index in data
    Parameters
    ----------
    :param index:        coordonate of point (list)
    :param data:         binary image or volume (ndarray)
    :param structure:    connectivity structure, default is full (list)
    Returns
    -------
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


def separate_blob(body_mask, cell_mask=None, min_dist=2, distances=[1, 1, 1]):
    """Attempt at Separating blobs using foreground pixels distance from background pixels.
    Parameters
    ----------
    mask : list, ndarray
        binary mask of the cells
    body : list, ndarray
        binary mask of the bodies
    min_dist : float
        minimum distance between coordinate to be seen as the center of a cell body
    distances : list, ndarray
        axis distances in order
    Returns
    -------
    body_labels_list : list
        list of cells body
    cell_labels_list : list
        list of cells mask
    """
    if cell_mask is None:
        cell_mask = body_mask.copy()
    
    # find 1 local max per cell
    distance = nd.distance_transform_edt(body_mask)
    tmp_coords = feature.peak_local_max(distance,
                                        footprint=np.ones((52, 40, 40)),
                                        labels=body_mask)
    
    # min_distance doesn't work with peak_local_max when using footprint
    # here's my version
    if len(tmp_coords) == 0:
        return [], []
    
    i = 1
    coords = [tmp_coords[0]]

    while i < len(tmp_coords):
        far_enough = True
        for co in coords:
            dist = utils.point_distance(co, tmp_coords[i], distances)

            if dist < min_dist:
                far_enough = False
                break

        if far_enough:
            coords.append(tmp_coords[i])

        i += 1

    coords = np.array(coords)
    
    # seperate the cells
    local_max = np.zeros(distance.shape, dtype=bool)
    local_max[tuple(coords.T)] = True
    markers = nd.label(local_max)[0]
    cell_labels = segmentation.watershed(-distance, markers, mask=cell_mask)
    regions = measure.regionprops(cell_labels)

    # restore cells with cytonemes
    cell_labels_list = []

    for rg in regions:
        cell_labels_list.append(cell_labels == rg.label)

    body_labels_list = []

    for cell in cell_labels_list:
        body_labels_list.append(np.logical_and(cell, body_mask))

    return cell_labels_list, body_labels_list