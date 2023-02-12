import numpy as np
import scipy.ndimage as nd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Replace or merge with seperate_blob implementations in carreno.utils.morphology
def seperate_blobs(x, min_dist=10, distances=[1, 1, 1]):
    """Separate blobs using watershed. Seeds are found using the foreground pixels distance from background pixels.
    Parameters
    ----------
    x : list, ndarray
        binary mask of blobs
    min_dist : float
        minimum distance between seeds for watershed
    distances : list, ndarray
        axis distances in order (TODO not used)
    Returns
    -------
    label : ndarray
        labelled blob
    """
    # find 1 local max per blob
    distance = nd.distance_transform_edt(x)
    coords = peak_local_max(distance,
                            min_distance=min_dist,
                            labels=x > 0)
    
    # seperate the cells
    local_max = np.zeros(distance.shape, dtype=bool)
    local_max[tuple(coords.T)] = True
    markers = nd.label(local_max)[0]
    label = watershed(-distance, markers, mask=x)

    return label


def associate_cytoneme(body_label, cyto_label):
    """Matches cytonemes with the nearest body.
    By using distance_transform_edt, we avoid checking each body voxel distance from each cyto voxel
    TODO express ambiguity
    Parameters
    ----------
    body_label : ndarray
        labelled bodies
    cyto_label : ndarray
        labelled cytonemes
    Returns
    -------
    association : [[]]
        list for each body containing a list of associated cytonemes. Axis 0 for body, axis 1 for associated cytonemes
    """
    # in case there is no cell body to match
    if not body_label.any():
        return []

    lv = 0.9   # low value (as long as it is < 1)
    hv = 1e10  # high value
    body_mask = body_label > 0
    body_dist = nd.distance_transform_edt(body_mask)
    body_dist[body_dist == 0] = lv

    # list of associated cytonemes per body
    association = []
    for i in range(body_label.max()):
        association.append([])

    # fill association
    for lb in range(1, cyto_label.max()):
        cyto = cyto_label == lb
        cyto_dist = nd.distance_transform_edt(np.logical_not(cyto))
        cyto_dist[cyto_dist == 0] = hv
        
        # closest point on a body TODO consider all min for handling cyto intersections
        depth, row, column = np.unravel_index(np.argmin(body_dist * cyto_dist), body_mask.shape)

        association[body_label[depth, row, column] - 1].append(lb)
    
    return association