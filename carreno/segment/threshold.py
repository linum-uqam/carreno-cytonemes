import numpy as np
from skimage.restoration import richardson_lucy, denoise_nl_means, estimate_sigma
from skimage import morphology, measure
from skimage.filters import threshold_otsu
import bm3d
from scipy import ndimage as nd
from carreno.utils.util import normalize

def primary_object(x, denoise=None, psf=None, sharpen=False):
    """Get mask of biggest object segmentation
    Parameters
    ----------
    x : list, ndarray
          array containing object(s)
    denoise : str, None
        denoising function before segmentation. Choices: 'bm' (block-matching) and 'nlm' (non-local mean)
    psf : ndarray
        another denoising option for applying richardson lucy filter
    sharpen : bool
        sharpens x before segmentation with maximum gaussian 2e derivative
    Returns
    -------
    mask : mask of object (ndarray)
    """
    # normalize between 0 and 1
    y = normalize(np.array(x), 0, 1)
    v_min = y.min()
    v_max = y.max()
    
    # denoising
    if psf is not None:
        y = richardson_lucy(y, psf, 20)
    
    if denoise == 'bm': # block-matching 3d
        y = bm3d.bm3d(y,
                      sigma_psd=30/255,
                      stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    elif denoise == 'nlm': # non-local mean
        sigma_est = np.mean(estimate_sigma(y))
        # incredibly slow (even with fast_mode)
        y = denoise_nl_means(y,
                             h=0.8*sigma_est,  # default is too low
                             sigma=sigma_est,
                             fast_mode=True,
                             preserve_range=True)
        
    # ostu threshold
    threshold = threshold_otsu(y)

    if sharpen:
        gaussian = np.maximum(
            nd.gaussian_filter(y, 1.5, 2),
            nd.gaussian_filter(y,   2, 2),
            nd.gaussian_filter(y,   4, 2)
        )
        gaussian = normalize(gaussian, v_min, threshold)
        
        details = np.abs(y - gaussian)
        
        """Doesn't really improve final result
        # details will be less represented near Z middle
        cos_distribution = np.abs(np.cos(np.linspace(0, math.pi, details.shape[0]))) + 1
        for i in range(len(cos_distribution)):
            details[i] = details[i] * cos_distribution[i]
        """
        
        y = y + details
        #y = np.clip(y, v_min, v_max)
                
    # flood-fill
    mask = y >= threshold
    fill = nd.binary_fill_holes(mask)
    
    # remove object smaller than main cell (risk removing arm)
    lb = measure.label(fill)
    rp = measure.regionprops(lb)
    sizes = sorted([j.area for j in rp])
    
    if len(sizes) > 1:  # method 1
        mask = morphology.remove_small_objects(lb, sizes[-2] + 1)
    else:               # method 2
        # BUG: for an unknown reason, this sometimes remove the cell, hence why method 1
        max_size = max([j.area for j in rp])
        mask = morphology.remove_small_objects(lb, max_size - 1)
    
    return mask