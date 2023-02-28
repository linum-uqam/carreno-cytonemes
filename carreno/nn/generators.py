# -*- coding: utf-8 -*-
import tensorflow as tf
import tifffile as tif
import numpy as np
from sklearn.utils import shuffle
from carreno.utils.util import is_2D, is_3D

def balanced_class_weights(labels):
    """ TODO make another script for sample weight creation?
    Give balanced weights for labels following this formula :
    (1 / class_instance) * (total_instance / nb_classes)
    Parameters
    ----------
    labels : [path]
        list of tiff volume paths
    Returns
    -------
    weights : [float]
        weight for each classes
    """
    # assume y is categorical (1 channel per class)
    weights = []
        
    list_of_volumes = labels
    if len(list_of_volumes) > 0:
        big_pile = tif.imread(list_of_volumes[0])
        for mask in list_of_volumes[1:]:
            big_pile += tif.imread(mask)

        nb_classes = big_pile.shape[-1]
        total = big_pile.sum()
        for i in range(nb_classes):
            class_instance = big_pile[..., i].sum()
            w_result = (1 / class_instance) * (total / nb_classes)
            weights.append(w_result)
            
    return weights


def get_volumes_slices(paths):
    """
    Get all slice index for all the volumes in list of paths
    Parameters
    ----------
    paths : [str]
        Paths to volumes to slice up
    Returns
    -------
    slices : [[str, int]]
        list of list containing volume names and slice indexes
    """
    slices = []

    for path in paths:
        with tif.TiffFile(path) as tif_file:
            # Wrong photometric in tiff format may cause issue when using pages
            #print('nb of pages', len(tif_file.pages))
            for i in range(len(tif_file.pages)):
                slices.append([path, i])
    
    return slices


def __augmentation(aug, x, y, w=None, noise=None):
    #https://medium.com/pytorch/multi-target-in-albumentations-16a777e9006e
    x_aug = None
    y_aug = None
    
    data = {'image' : x,
            'mask'  : y}
    if not w is None:
        data['weight'] = w

    if not aug is None:
        data = aug(**data)

    y_aug = data['mask']
    w_aug = data['weight']

    if noise is None:
        x_aug = data['image']
    else:
        noisy_data = {'image': data['image']}
        noisy_data = noise(**noisy_data)
        x_aug = noisy_data['image']
    
    # model architecture requires a color channel axis
    if (is_2D(x.shape) and x_aug.ndim < 3) or (is_3D(x.shape) and x_aug.shape.ndim < 4):
        x_aug = np.expand_dims(x_aug, axis=-1)

    return x_aug, y_aug, w_aug


class volume_slice_generator(tf.keras.utils.Sequence):
    def __init__(self, vol, label, size=1, augmentation=None, noise=None, shuffle=True, weight=None):
        """
        Data generator for 2D model training
        Parameters
        ----------
        img : [[str, int]]
            path to volume followed by volume slice index
        label : [[str, int]]
            path to volume followed by volume slice index
        size : int
            batch size
        augmentation : albumentations.Compose
            composition of transformations applied to both X and Y
        noise : albumentations.Compose
            composition of transformations applied only to X
        shuffle : bool
            whether we should shuffle after each epoch
        weight : str, None
            apply sample weight for unbalanced dataset
            -"balanced" : (1/instances) * (total/nb_classes)
        """
        self.x = vol
        self.y = label
        self.size = size
        self.aug = augmentation
        self.noise = noise
        self.shuffle = shuffle
        if not weight is None:
            self.w = balanced_class_weights(list(set([path for path, slc in self.y])))
        if len(self.x) > 0:
            self.__missing_color_ch = len(tif.imread(self.x[0][0]).shape) < 4
    
    def __len__(self):
        # number of batch (some img might not be included in the epoch)
        return len(self.y) // self.size
        
    def __getitem__(self, idx):
        data_i = idx * self.size
        
        # instead of reading entire volume, reads only 1 slice using TiffFile
        with tif.TiffFile(self.x[data_i][0]) as x_info:
            x_batch = np.zeros((self.size, * x_info.pages[0].shape))
        with tif.TiffFile(self.y[data_i][0]) as y_info:
            y_batch = np.zeros((self.size, * y_info.pages[0].shape))

        # fill batches
        for i in range(self.size):
            x_path, slc = self.x[data_i]
            y_path, __  = self.y[data_i]
            
            data = {}
            with tif.TiffFile(x_path) as x_info:
                data['image'] = x_info.pages[slc].asarray()
            with tif.TiffFile(y_path) as y_info:
                data['mask'] = y_info.pages[slc].asarray()
            
            if not self.aug is None:
                data = self.aug(**data)
            
            y_batch[i] = data['mask']

            if self.noise is None:
                x_batch[i] = data['image']
            else:
                noisy_data = {'image': data['image']}
                noisy_data = self.noise(**noisy_data)
                x_batch[i] = noisy_data['image']

            data_i += 1
        
        # model architecture requires a color channel axis
        if self.__missing_color_ch:
            x_batch = np.expand_dims(x_batch, axis=3)
        
        batch = [x_batch, y_batch]
        if hasattr(self, "w"):
            # weight ndarray doesn't need the color channel axis
            sample_weights = np.zeros(y_batch.shape[:-1])
            
            for i in range(y_batch.shape[-1]):
                sample_weights[y_batch[..., i] == True] = self.w[i]
            
            batch.append(sample_weights)

        self.batch = batch  # lets us get the latest batch in callbacks

        return batch
        
    def on_epoch_end(self):
        if self.shuffle:
            self.x, self.y = shuffle(self.x, self.y)


class volume_generator(tf.keras.utils.Sequence):
    def __init__(self, vol, label, size=1, augmentation=None, shuffle=True, weight=None):
        """
        Data generator for 3D model training
        Parameters
        ----------
        vol : [str]
            path to input volume
        label : [str]
            path to label volume
        augmentation : volumentations.Compose
            composition of transformations
        size : int
            batch size
        shuffle : bool
            whether we should shuffle after each epoch
        weight : str, None
            apply sample weight for unbalanced dataset
            -"balanced" : (1/instances) * (total/nb_classes)
        """
        self.x = vol
        self.y = label
        self.size = size
        self.aug = augmentation
        self.shuffle = shuffle
        self.x_shape = []
        self.y_shape = []
        if not weight is None:
            self.w = balanced_class_weights(self.y)
        if len(self.x) > 0:
            # instead of reading entire volume, reads only 1 slice using TiffFile
            x_info = tif.TiffFile(self.x[0])
            y_info = tif.TiffFile(self.y[0])
            self.x_shape = tuple([len(x_info.pages)] + list(x_info.pages[0].shape))
            self.y_shape = tuple([len(y_info.pages)] + list(y_info.pages[0].shape))
            self.__missing_color_ch = len(self.x_shape) < 5
    
    def __len__(self):
        # number of batch (some img might not be included in the epoch)
        return len(self.y) // self.size
        
    def __getitem__(self, idx):
        data_i = idx * self.size
        x_batch = np.zeros((self.size, *self.x_shape))
        y_batch = np.zeros((self.size, *self.y_shape))
        
        # fill batches
        for i in range(self.size):
            x_path = self.x[data_i]
            y_path = self.y[data_i]
            x_info = tif.imread(x_path)
            y_info = tif.imread(y_path)
            data = {
                'image': x_info,
                'mask' : y_info
            }

            if not self.aug is None:
                data = self.aug(**data)
            
            x_batch[i] = data['image']
            y_batch[i] = data['mask']
            
            data_i += 1
        
        # model architecture requires a color channel axis
        if self.__missing_color_ch:
            x_batch = np.expand_dims(x_batch, axis=4)
        
        batch = [x_batch, y_batch]
        if hasattr(self, "w"):
            # weight ndarray doesn't need the color channel axis
            sample_weights = np.zeros(y_batch.shape[:-1])
            
            for i in range(y_batch.shape[-1]):
                sample_weights[y_batch[..., i] == True] = self.w[i]
            
            batch.append(sample_weights)

        self.batch = batch  # lets us get the latest batch in callbacks

        return batch
    
    def on_epoch_end(self):
        if self.shuffle:
            self.x, self.y = shuffle(self.x, self.y)
    