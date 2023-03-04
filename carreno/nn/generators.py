# -*- coding: utf-8 -*-
import tensorflow as tf
import tifffile as tif
import numpy as np
from sklearn.utils import shuffle
from carreno.utils.util import is_2D, is_3D


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


def augment(aug, x, y, w=None, noise=None): 
    x_aug = None
    y_aug = None
    w_aug = None
    w_param = not w is None

    data = {'image' : x,
            'mask'  : y}
    if w_param:
        data['weight'] = w

    if not aug is None:
        data = aug(**data)

    y_aug = data['mask']
    if w_param:
        w_aug = data['weight']

    if noise is None:
        x_aug = data['image']
    else:
        # add noise to input
        noisy_data = {'image': data['image']}
        noisy_data = noise(**noisy_data)
        x_aug = noisy_data['image']

    batch = [x_aug, y_aug] if not w_param else [x_aug, y_aug, w_aug]

    return batch


class volume_slice_generator(tf.keras.utils.Sequence):
    def __init__(self, vol, label, weight=None, size=1, augmentation=None, noise=None, shuffle=True, nb_color_ch=1):
        """
        Data generator for 2D model training
        Parameters
        ----------
        img : [[str, int]]
            path to volume followed by volume slice index
        label : [[str, int]]
            path to volume followed by volume slice index
        weight : [[str, int]]
            path to volume followed by volume slice index
        size : int
            batch size
        augmentation : albumentations.Compose
            composition of transformations applied to both inputs and labels
        noise : albumentations.Compose
            composition of transformations applied only to inputs
        shuffle : bool
            whether we should shuffle after each epoch
        nb_color_ch : int
            number of color channels if there isn't any (mainly for grayscale)
        """
        self.x = vol
        self.y = label
        self.w = weight
        self.size = size
        self.aug = augmentation
        self.noise = noise
        self.shuffle = shuffle
        self.x_shape = []
        self.y_shape = []
        self.nb_color_ch = nb_color_ch
        if len(self.x) > 0:
            # instead of reading entire volume, reads only 1 slice using TiffFile
            with tif.TiffFile(self.x[0][0]) as x_info:
                self.x_shape = (x_info.pages[0].shape)
            with tif.TiffFile(self.y[0][0]) as y_info:
                self.y_shape = (y_info.pages[0].shape)
            self.__missing_color_ch = len(self.x_shape) < 3
    
    def __len__(self):
        # number of batch (some img might not be included in the epoch)
        return len(self.y) // self.size
        
    def __getitem__(self, idx):
        data_i = idx * self.size
        
        # instead of reading entire volume, reads only 1 slice using TiffFile
        x_batch = np.zeros((self.size, *self.x_shape))
        y_batch = np.zeros((self.size, *self.y_shape))
        w_batch = None
        if not self.w is None:
            w_batch = x_batch.copy()

        # fill batches
        for i in range(self.size):
            x_path, slc = self.x[data_i]
            x_info = tif.TiffFile(x_path)
            x = x_info.pages[slc].asarray()  # load only 1 slice
            x_info.close()  # close opened file

            y_path, __  = self.y[data_i]
            y_info = tif.TiffFile(y_path)
            y = y_info.pages[slc].asarray()
            y_info.close()
            
            w = None
            if not w_batch is None:
                w_path, __  = self.w[data_i]
                w_info = tif.TiffFile(w_path)
                w = w_info.pages[slc].asarray()
                w_info.close()
                
            data = augment(aug=self.aug,
                           x=x,
                           y=y,
                           w=w,
                           noise=self.noise)
            
            # fill batch w augmentation
            x_batch[i], y_batch[i] = data[:2]
            if not self.w is None:
                w_batch[i] = data[2]
            
            data_i += 1

        if self.__missing_color_ch:
            # add color channel
            x_batch = np.stack([x_batch] * self.nb_color_ch, axis=-1)

        batch = [x_batch, y_batch] if self.w is None else [x_batch, y_batch, w_batch]

        return batch
        
    def on_epoch_end(self):
        if self.shuffle:
            self.x, self.y = shuffle(self.x, self.y)


class volume_generator(tf.keras.utils.Sequence):
    def __init__(self, vol, label, size=1, augmentation=None, noise=None, shuffle=True, weight=None, nb_color_ch=1):
        """
        Data generator for 3D model training
        Parameters
        ----------
        vol : [str]
            path to input volume
        label : [str]
            path to label volume
        weight : [[str, int]]
            path to volume followed by volume slice index
        size : int
            batch size
        augmentation : volumentations.Compose
            composition of transformations applied to both inputs and labels
        noise : volumentations.Compose
            composition of transformations applied only to inputs
        shuffle : bool
            whether we should shuffle after each epoch
        nb_color_ch : int
            number of color channels if there isn't any (mainly for grayscale)
        """
        self.x = vol
        self.y = label
        self.w = weight
        self.size = size
        self.aug = augmentation
        self.noise = noise
        self.shuffle = shuffle
        self.x_shape = []
        self.y_shape = []
        self.nb_color_ch = nb_color_ch
        if len(self.x) > 0:
            # instead of reading entire volume, reads only 1 slice using TiffFile
            with tif.TiffFile(self.x[0]) as x_info:
                self.x_shape = tuple([len(x_info.pages)] + list(x_info.pages[0].shape))
            with tif.TiffFile(self.y[0]) as y_info:
                self.y_shape = tuple([len(y_info.pages)] + list(y_info.pages[0].shape))
            self.__missing_color_ch = len(self.x_shape) < 4
    
    def __len__(self):
        # number of batch (some img might not be included in the epoch)
        return len(self.y) // self.size
        
    def __getitem__(self, idx):
        data_i = idx * self.size
        x_batch = np.zeros((self.size, *self.x_shape))
        y_batch = np.zeros((self.size, *self.y_shape))
        w_batch = None
        if not self.w is None:
            w_batch = x_batch.copy()

        # fill batches
        for i in range(self.size):
            x = tif.imread(self.x[data_i])  # files are closed after reading into ndarray
            y = tif.imread(self.y[data_i])
            w = None if self.w is None else tif.imread(self.w[data_i])

            data = augment(aug=self.aug,
                                x=x,
                                y=y,
                                w=w,
                                noise=self.noise)
            
            # fill batch w augmentation
            x_batch[i], y_batch[i] = data[:2]
            if not self.w is None:
                w_batch[i] = data[2]
            
            data_i += 1
        
        # model architecture requires a color channel axis
        if self.__missing_color_ch:
            x_batch = np.stack([x_batch] * self.nb_color_ch, axis=-1)
        
        batch = [x_batch, y_batch] if self.w is None else [x_batch, y_batch, w_batch]
        
        return batch
    
    def on_epoch_end(self):
        if self.shuffle:
            self.x, self.y = shuffle(self.x, self.y)
    