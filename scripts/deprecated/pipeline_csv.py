# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import tifffile as tif
import numpy as np
import bm3d
from skimage.restoration import richardson_lucy, denoise_nl_means, estimate_sigma
from skimage.morphology import binary_opening, binary_closing, skeletonize_3d
from skimage.transform import resize
from scipy.stats import median_absolute_deviation
from scipy import ndimage as nd
from pathlib import Path


# local imports
import utils
import carreno.utils.array
import carreno.utils.morphology
import carreno.io.tifffile
import carreno.processing.patches
import carreno.processing.categorical
import carreno.threshold.threshold
import carreno.cell.path

model_name  = "slfspv_1e-05-untrn_unet2D-4-64-0.3-relu-VGG16.h5"  # model filename or None
volume_name = "slik6.tif"
psf_name    = None  #"Averaged PSF.tif"  # denoising option for applying richardson lucy filter (put None if not wanted)
denoise     = None  # denoise volume ("bm", "nlm", None)
sharpen     = 0  # sharpen factor for volume before segmentation using maximum gaussian 2e derivative
fname       = lambda path : path.split("/")[-1].split(".", 1)[0]  # get filename without extension


def main(verbose=0):
    config   = utils.get_config()
    vol_path = os.path.join(config['VOLUME']['input']   , volume_name)
    csv_path = os.path.join(config['BASE']['output']    , "model-{}_vol-{}.csv".format(fname(model_name) if model_name else None,
                                                                                       fname(volume_name)))

    # find axes distances between coordinates
    distances = [1] * 3
    try:
        # assuming unit is um
        distances = carreno.io.tifffile.metadata(vol_path)["axe_dist"]
        # convert scale to picometer like in imagej since selem is scaled for it
        distances = np.array(distances) * 1e6
    except:
        # if we don't have axis info, use default distances
        distances = np.array([0.26, 0.1201058, 0.1201058])

    # normalize between 0 and 1
    vol = tif.imread(vol_path)
    vol = carreno.utils.array.normalize(vol, 0, 1)
    
    #############
    # DENOISING #
    #############
    
    if psf_name is not None:
        psf_path = os.path.join(config['BASE']['psf'], psf_name)
        psf      = tif.imread(psf_path)
        vol      = richardson_lucy(vol, psf, 20)
    
    if denoise == 'bm':     # block-matching 3d
        # refer to https://www.mathworks.com/help/wavelet/ref/wnoisest.html
        # for sigma_psd estimation
        vol = bm3d.bm3d(vol,
                        sigma_psd=median_absolute_deviation(vol)/0.6745,
                        stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    elif denoise == 'nlm':  # non-local mean
        sigma_est = np.mean(estimate_sigma(y))
        # incredibly slow (even with fast_mode)
        y = denoise_nl_means(y,
                             h=0.8*sigma_est,  # default is too low
                             sigma=sigma_est,
                             fast_mode=True,
                             preserve_range=True)

    if sharpen > 0:
        gaussian = np.maximum(
            nd.gaussian_filter(vol, 1.5, 2),
            nd.gaussian_filter(vol,   2, 2),
            nd.gaussian_filter(vol,   4, 2)
        )
        
        details = np.abs(vol - gaussian)
        vol = vol + (details * sharpen)
        vol = carreno.utils.array.normalize(vol, 0, 1)
    
    ################
    # SEGMENTATION #
    ################
    
    # sphere structure element for morphologic operations
    sphere = carreno.utils.morphology.create_sphere(2, distances)
    if model_name:
        # CNN
        mod_path = os.path.join(config['TRAINING']['output'], model_name) if not model_name is None else None
        model  = tf.keras.models.load_model(mod_path, compile=False)
        model_input = model.layers[0].input.shape[1:]
        ncolor = carreno.utils.array.nb_color_channel(model_input)
        ndim   = carreno.utils.array.ndim_for_pixel(model_input)

        # make grayscale volume match requested nb of color channel for model input layer
        c_vol = np.stack([vol] * ncolor, axis=-1)

        pred = None
        if ndim == 2:  # 2D model
            stride = [1] + config['PREPROCESS']['stride'][1:]
            pred = carreno.processing.patches.volume_pred_from_img(model,
                                                                   c_vol,
                                                                   stride=stride,
                                                                   weight=None)
        else:          # 3D model
            pred = carreno.processing.patches.volume_pred_from_vol(model,
                                                                   c_vol,
                                                                   stride=config['PREPROCESS']['stride'],
                                                                   weight=None)

        # keep volume shape which might have changed with patchify
        pred = resize(pred,
                      output_shape=vol.shape[:3],
                      order=1,
                      preserve_range=True,
                      anti_aliasing=False)

        # hard labels (binarization of labels) TODO possible optimisation of threshold
        segmentation = carreno.processing.categorical.categorical_multiclass(pred)

        cyto_m = segmentation[..., 1]
        body_m = segmentation[..., 2]

        # post-processing for body uniformity
        body_m = binary_opening(binary_closing(body_m, selem=sphere), selem=sphere)
    else:
        # THRESHOLDING
        
        # get cell(s) mask
        cell_m = carreno.threshold.threshold.primary_object(vol, size=40**3)

        # removes cytonemes from cell and smooth body edges
        body_m = binary_opening(binary_closing(cell_m, selem=sphere), selem=sphere)

        # get cell(s) cytoneme mask
        cyto_m = cell_m.copy()
        cyto_m[body_m == 1] = 0
    
    #################
    # CELL ANALYSIS #
    #################
    
    carreno.cell.path.extract_metric(body_m, cyto_m, csv_path, distances=distances)


if __name__ == "__main__":
    main(verbose=1)