# -*- coding: utf-8 -*-
import numpy as np
from skimage.restoration import richardson_lucy, denoise_nl_means, estimate_sigma
from skimage import morphology, measure
from skimage.filters import threshold_otsu
import bm3d
from scipy import ndimage as nd


def primary_object(x, size=None):
    """Get mask of big objects in the given volume
    Parameters
    ----------
    x : array-like
        Volume for binary segmentation
    size : float
        Minimum size for objects. No size returns the biggest object
    Returns
    -------
    mask : mask of object (ndarray)
    """
    # ostu threshold
    threshold = threshold_otsu(x)
                
    # flood-fill
    mask = x >= threshold
    fill = nd.binary_fill_holes(mask)
    
    # remove object smaller than main cell (risk removing arm)
    lb = measure.label(fill)
    rp = measure.regionprops(lb)
    sizes = sorted([j.area for j in rp])
    
    if size:
        mask = morphology.remove_small_objects(lb, size)
    elif len(sizes) > 1:  # method 1
        mask = morphology.remove_small_objects(lb, sizes[-2] + 1)
    else:                 # method 2
        # BUG: for an unknown reason, this sometimes remove the biggest object, hence why method 1
        max_size = max([j.area for j in rp] + [1])  # +[1] just in case theres no objects
        mask = morphology.remove_small_objects(lb, max_size - 1)
    
    return mask