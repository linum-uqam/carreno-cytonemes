# -*- coding: utf-8 -*-
import numpy as np
import tifffile as tif
import skimage
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.restoration
import skimage.feature
import skimage.segmentation
import scipy.ndimage as nd

import carreno.utils.morphology
import carreno.utils.array
import carreno.processing.categorical
import carreno.processing.patches
from carreno.cell.path import extract_metric


class Pipeline:
    def __init__(self, distances=(0.26, 0.1201058, 0.1201058)):
        self.distances = distances
    
    def train(self, x, y):
        raise NotImplementedError

    def restore(self, x, psf=None, iteration=50, nlm_size=7, nlm_dist=11, r=50, amount=8, md_size=(3,5,5), butterworth=None):
        """
        Parameters
        ----------
        x : ndarray
            volume to restore
        psf : ndarray
            psf volume
        nlm_size : int
            patch_size
            See https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_nl_means
        nlm_dist : float
            patch_distance
            See https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_nl_means
        r : int
            sharpening radius
        amount : float
            The details will be amplified with this factor. The factor could be 0
            or negative. Typically, it is a small positive number, e.g. 1.0.
        md_size : list
            Dimension for median filter (int, int, int)
        butterworth : list
            butterworth params (low, high, shift) using ratio from 0 to 1
            see carreno.utils.array.butterworth
        Returns
        -------
        median : ndarray
            restored volume
        """
        denoise = x.copy()

        if not psf is None:
            denoise = skimage.restoration.richardson_lucy(denoise, psf, iteration)

        if nlm_size > 0:
            # NLM is slow (3-5 minutes) and less accurate than with PSF
            sigma_est = skimage.restoration.estimate_sigma(denoise)
            denoise = skimage.restoration.denoise_nl_means(denoise,
                                                           patch_size=nlm_size,
                                                           patch_distance=nlm_dist,
                                                           h=0.8*sigma_est,  # recommend slightly less than standard deviation
                                                           sigma=sigma_est,
                                                           fast_mode=True,   # cost more memory
                                                           preserve_range=True)

        unsharp = skimage.filters.unsharp_mask(denoise, r, amount, preserve_range=True)
        
        median = unsharp
        if (np.array(md_size) > 1).any():
            median = nd.median_filter(median, size=md_size)
        
        # weight the first axis to ignore blurs
        if butterworth:
            params = (np.array(butterworth) * x.shape[0]).astype(int)
            btw = carreno.utils.array.butterworth(x.shape[0], *params)
            median = median * btw[:, np.newaxis, np.newaxis]

        return median

    def segmentation(self, x):
        raise NotImplementedError
    
    def analyse(self, pred, output):
        x = carreno.processing.categorical.categorical_to_sparse(pred)
        extract_metric(body_m=x==3, cyto_m=x==2, csv_output=output, distances=self.distances)
        return


class Threshold(Pipeline):
    def __init__(self, distances=(0.26, 0.1201058, 0.1201058)):
        super().__init__(distances=distances)

    def segmentation(self, x, ro=1.0, rc=1.5):
        """
        Parameters
        ----------
        x : ndarray
            denoised volume to segment
        ro : float
            radius for opening (to seperate classes)
        rc : float
            radius for closing (to ease the hole filling)
        Returns
        -------
        cls : ndarray
            volume segmented between background, cytoneme and body classes
        """
        otsu = x > skimage.filters.threshold_otsu(x, nbins=256)
        def separate_cls(x, selem):
            body = skimage.morphology.binary_opening(x, selem).astype(np.uint8)
            cyto = ((x - body) > 0).astype(np.uint8)
            return body * 2 + cyto
        def seperate_cls_w_closing(x, ro, rc, distances=[1,1,1]):
            # structuring element in 2D since third axis is inconsistent
            sphere = carreno.utils.morphology.create_sphere(ro, distances)
            disk = np.expand_dims(np.amax(sphere, axis=0), axis=0)  # disk instead of sphere
            y = separate_cls(x, selem=disk)
            # fill bodies
            body = y == 2
            sphere = carreno.utils.morphology.create_sphere(rc, distances)
            dilate = skimage.morphology.binary_dilation(body, sphere)
            for i in range(dilate.shape[0]):
                dilate[i] = nd.morphology.binary_fill_holes(dilate[i])
            closing = skimage.morphology.binary_erosion(dilate, sphere)
            y[closing] = 2
            # try seperating classes again
            y = separate_cls(y > 0, selem=disk)
            return y
        pred = seperate_cls_w_closing(otsu, ro=ro, rc=rc, distances=self.distances)
        pred = carreno.processing.categorical.sparse_to_categorical(pred+1, 3)
        # in case values are missing, we default to cell body
        missing = 1 - pred.sum(axis=-1)
        pred[..., 2] = pred[..., 2] + missing
        return pred


class UNet2D(Pipeline):
    def __init__(self, model2D, distances=(0.26, 0.1201058, 0.1201058)):
        """
        Parameters
        ----------
        model2D : model
            UNet 3D keras model
        """
        super().__init__(distances=distances)
        self.model2D = model2D

    def segmentation(self, x, stride, weight=None):
        """
        Load UNet 2D model and predict volume in patches
        Parameters
        ----------
        x : ndarray
            Volume to predict
        stride : [int, int]
            Stride between patches, defines interlaps
        weight : None, ndarray
            Weight matrix with the same shape as patches
        Returns
        -------
        pred : ndarray
            Prediction of x
        """
        pred = carreno.processing.patches.volume_pred_from_img(
            self.model2D,
            x if len(x.shape) >= 4 else np.expand_dims(x, axis=-1),  # add color axis
            [1] + stride,
            None if weight is None else np.expand_dims(weight, axis=0))
        # in case values are missing, we default to cell body
        missing = 1 - pred.sum(axis=-1)
        pred[..., 2] = pred[..., 2] + missing
        return pred # remove batch size axis


class UNet3D(Pipeline):
    def __init__(self, model3D, distances=(0.26, 0.1201058, 0.1201058)):
        """
        Parameters
        ----------
        model3D : model
            UNet 3D keras model
        """
        super().__init__(distances=distances)
        self.model3D = model3D

    def segmentation(self, x, stride, weight=None):
        """
        Load UNet 3D model and predict volume in patches
        Parameters
        ----------
        x : ndarray
            Volume to predict
        stride : [int, int, int]
            Stride between patches, defines interlaps
        weight : None, ndarray
            Weight matrix with the same shape as patches
        Returns
        -------
        pred : ndarray
            Prediction of x
        """
        pred = carreno.processing.patches.volume_pred_from_vol(
            self.model3D,
            x if len(x.shape) >= 4 else np.expand_dims(x, axis=-1),  # add color axis
            stride,
            None if weight is None else weight)
        # in case values are missing, we default to cell body
        missing = 1 - pred.sum(axis=-1)
        pred[..., 2] = pred[..., 2] + missing
        return pred


if __name__ == '__main__':
    import tifffile as tif
    import matplotlib.pyplot as plt

    pipeline = Threshold()
    psf = tif.imread("data/psf/Averaged PSF.tif")

    vol = tif.imread("data/dataset/input/slik4.tif")
    pred = pipeline.segmentation(vol, psf)

    zs = np.linspace(0, vol.shape[0], 9)
    plt.subplot(331)
    plt.imshow(pred[zs[0]])
    plt.subplot(332)
    plt.imshow(pred[zs[1]])
    plt.subplot(333)
    plt.imshow(pred[zs[2]])
    plt.subplot(334)
    plt.imshow(pred[zs[3]])
    plt.subplot(335)
    plt.imshow(pred[zs[4]])
    plt.subplot(336)
    plt.imshow(pred[zs[5]])
    plt.subplot(337)
    plt.imshow(pred[zs[6]])
    plt.subplot(338)
    plt.imshow(pred[zs[7]])
    plt.subplot(339)
    plt.imshow(pred[zs[8]])
    plt.show()