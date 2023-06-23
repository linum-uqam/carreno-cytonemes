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


class Pipeline:
    def __init__(self, distances=(0.26, 0.1201058, 0.1201058)):
        self.distances = distances
    
    def train(self, x, y):
        raise NotImplementedError

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
            Minimum area for body in nm
        s2 : float
            Minimum area for region in nm
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
            min_px = size / px_superficie
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
    
    def train(self, x, y):
        raise NotImplementedError

    def segmentation(self, x, psf=None, low=20, high=15, r1=0.5, r2=1):
        # denoise
        if psf is None:
            # NLM is slow (3-5 minutes) and less accurate than with PSF
            sigma_est = skimage.restoration.estimate_sigma(x)
            denoise = skimage.restoration.denoise_nl_means(x,
                                                           patch_size=7,
                                                           patch_distance=7,
                                                           h=0.8*sigma_est,  # recommend slightly less than standard deviation
                                                           sigma=sigma_est,
                                                           fast_mode=True,   # cost more memory
                                                           preserve_range=True)
        else:
            denoise = skimage.restoration.richardson_lucy(x, psf, iterations=50)

        def butterworth(n, low, high):
            """
            Butterworth distribution in 1D
            Parameters
            ----------
            n : int
                length of the distribution
            low : float
                low values boundaries
            high : float
                high values boundaries
            Returns
            -------
            n : ndarray
                distribution values in a list
            """
            u = np.linspace(0, n, n)
            D = np.sqrt((u-n/2)**2)
            h = 1 / (1 + (D/low)**(2*high))
            return h

        # weight the first axis to ignore blurs
        btw = butterworth(x.shape[0], low, high)
        weighted = denoise * btw[:, np.newaxis, np.newaxis]

        amount = 10
        unsharp = skimage.filters.unsharp_mask(weighted, 50, amount=amount)

        median = nd.median_filter(unsharp, size=[1,3,3])

        otsu = median > skimage.filters.threshold_otsu(median, nbins=256)
        
        def separate_cls(x, selem, distances=[1,1,1]):
            body = skimage.morphology.binary_opening(x, selem).astype(np.uint8)
            cyto = ((x - body) > 0).astype(np.uint8)
            return body * 2 + cyto
        
        def seperate_cls_w_closing(x, r1, r2, distances=[1,1,1]):
            # structuring element in 2D since third axis is inconsistent
            sphere = carreno.utils.morphology.create_sphere(r1, distances)
            disk = np.expand_dims(np.amax(sphere, axis=0), axis=0)  # disk instead of sphere
            y = separate_cls(x, selem=disk, distances=distances)

            # fill bodies
            body = y == 2
            sphere = carreno.utils.morphology.create_sphere(r2, distances)
            dilate = skimage.morphology.binary_dilation(body, selem=sphere)
            for i in range(dilate.shape[0]):
                dilate[i] = nd.morphology.binary_fill_holes(dilate[i])
            closing = skimage.morphology.binary_erosion(dilate, selem=sphere)
            y[closing] = 2

            # try seperating classes again
            y = separate_cls(y > 0, selem=disk, distances=distances)

            return y

        cls = seperate_cls_w_closing(otsu, r1=r1, r2=r2, distances=self.distances)

        return cls


class UNet2D(Pipeline):
    def __init__(self, distances=(0.26, 0.1201058, 0.1201058)):
        super().__init__(distances=distances)
        self.model2D = None
    
    def train(self, x, y):
        raise NotImplementedError

    def segmentation(self, x):
        raise NotImplementedError
    
    def analyse(self, p):
        raise NotImplementedError


class UNet3D(UNet2D):
    def __init__(self, distances=(0.26, 0.1201058, 0.1201058)):
        super().__init__(distances=distances)
        self.model3D = None
    
    def train(self, x, y):
        raise NotImplementedError

    def segmentation(self, x):
        raise NotImplementedError
    
    def analyse(self, p):
        raise NotImplementedError


if __name__ == '__main__':
    import tifffile as tif
    import matplotlib.pyplot as plt

    pipeline = Threshold()
    psf = tif.imread("data/psf/Averaged PSF.tif")

    vol = tif.imread("data/dataset/input/slik4.tif")
    pred = pipeline.segmentation(vol, psf)
    #plt.imshow(np.mean(pred, axis=0))
    plt.subplot(331)
    plt.imshow(pred[0])
    plt.subplot(332)
    plt.imshow(pred[5])
    plt.subplot(333)
    plt.imshow(pred[10])
    plt.subplot(334)
    plt.imshow(pred[15])
    plt.subplot(335)
    plt.imshow(pred[20])
    plt.subplot(336)
    plt.imshow(pred[25])
    plt.subplot(337)
    plt.imshow(pred[30])
    plt.subplot(338)
    plt.imshow(pred[35])
    plt.subplot(339)
    plt.imshow(pred[38])
    plt.show()