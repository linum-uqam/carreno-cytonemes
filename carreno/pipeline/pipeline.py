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
import tensorflow as tf

import carreno.utils.morphology
import carreno.utils.array
import carreno.processing.categorical
import carreno.processing.patches


class Pipeline:
    def __init__(self, distances=(0.26, 0.1201058, 0.1201058)):
        self.distances = distances
    
    def train(self, x, y):
        raise NotImplementedError

    def restore(self, x, psf=None, iteration=50, nlm_size=7, nlm_dist=11, r=50, amount=10, md_size=(1,3,3), butterworth=None):
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
    
    def postprocessing(self, pred, s1=1000, s2=100):
        """
        Corrections to prediction :
        1. Convert small bodies into cytonemes
        2. Removes small regions
        Parameters
        ----------
        pred : ndarray
            Prediction volume
        s1 : float
            Minimum area for body in um
        s2 : float
            Minimum area for region in um
        Returns
        -------
        pred : ndarray
            Updated prediction volume
        """
        process = pred.copy()

        def get_small_obj(x, size, distances=[1,1,1]):
            lb = skimage.measure.label(x)
            px_superficie = np.prod(distances)
            rp = skimage.measure.regionprops(lb)
            y = np.zeros_like(x)
            min_px = size * px_superficie
            for p in rp:
                if p.area <= min_px:
                    y[lb == p.label] = 1
            return y
        
        # convert small bodies to cyto
        process[get_small_obj(pred[..., 2], s1, distances=self.distances)] = [0,1,0]

        # remove cytonemes that are too small
        process = np.logical_and(pred, np.logical_not(get_small_obj(pred, s2, distances=self.distances)))

        return process

    def analyse(self, pred):
        def seperate_blobs(x, min_dist=5, num_peaks=np.inf, distances=[1, 1, 1]):
            """Separate blobs using watershed. Seeds are found using the foreground pixels distance from background pixels.
            Parameters
            ----------
            x : list, ndarray
                binary mask of blobs
            min_dist : float
                minimum distance between seeds for watershed
            distances : list, ndarray
                axis distances in order
            Returns
            -------
            label : ndarray
                labelled blob
            """
            y = x.copy()

            # find 1 local max per blob
            distance = nd.distance_transform_edt(y)
            min_dist = int(min_dist / min(distances))

            coords = skimage.feature.peak_local_max(distance,
                                                    min_distance=min_dist,
                                                    num_peaks=num_peaks,
                                                    labels=y)

            # seperate the cells
            local_max = np.zeros(distance.shape, dtype=bool)
            local_max[tuple(coords.T)] = True
            markers = nd.label(local_max)[0]
            label = skimage.segmentation.watershed(-distance, markers, mask=y)

            return label
        
        n_peaks = int((pred == 2).sum() // (500 / np.prod(self.distances)))
        bodies = seperate_blobs(pred == 2, min_dist=3, num_peaks=n_peaks, distances=self.distances)
        raise NotImplementedError


class Threshold(Pipeline):
    def __init__(self, distances=(0.26, 0.1201058, 0.1201058)):
        super().__init__(distances=distances)

    def segmentation(self, x, ro=0.5, rc=1):
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

        cls = seperate_cls_w_closing(otsu, ro=ro, rc=rc, distances=self.distances)
        cls = carreno.processing.categorical.sparse_to_categorical(cls+1, 3)

        return cls


class UNet2D(Pipeline):
    def __init__(self, model2D, distances=(0.26, 0.1201058, 0.1201058)):
        """
        Parameters
        ----------
        model : str, Path
            Path to UNet 2D model
        """
        super().__init__(distances=distances)
        self.model2D = tf.keras.models.load_model(model2D, compile=False)

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
            np.expand_dims(x, axis=-1),  # add color axis for grayscale img
            [1] + stride,
            None if weight is None else np.expand_dims(weight, axis=0))
        return np.squeeze(pred, axis=-1) # remove batch size axis


class UNet3D(UNet2D):
    def __init__(self, model3D, distances=(0.26, 0.1201058, 0.1201058)):
        """
        Parameters
        ----------
        model3D : str, Path
            Path to UNet 3D model
        """
        super().__init__(distances=distances)
        self.model3D = tf.keras.models.load_model(model3D, compile=False)

    def segmentation(self, x, stride, weight):
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
            x,
            stride,
            weight)
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