import numpy as np
import scipy.ndimage as nd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


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

    # distances from cells per different bodies over the axis 0
    body_masks = [body_label == i for i in range(1, body_label.max() + 1)]
    body_dists = np.array([nd.distance_transform_edt(~bm) for bm in body_masks])

    # list of associated cytonemes per body
    association = []
    for i in range(body_label.max()):
        association.append([])

    # fill association
    for lb in range(1, cyto_label.max() + 1):
        cyto = cyto_label == lb
        cyto = np.stack([cyto] * body_dists.shape[0], axis=0)

        # considering only the cyto coordinates, check which has the minimum distance to a cell body
        # get the body_dists axis 0 of this minimum to find which body is the closest
        min_dist = np.where(cyto)[0][body_dists[cyto].argmin()]

        association[min_dist].append(lb)
    
    return association