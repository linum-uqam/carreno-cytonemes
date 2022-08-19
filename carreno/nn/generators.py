# -*- coding: utf-8 -*-
import tensorflow as tf
import tifffile as tif
import numpy as np
from sklearn.utils import shuffle

class volume_slice_generator(tf.keras.utils.Sequence):
    def __init__(self, img, label, size=1, augmentation=None, shuffle=True, weight=None):
        """
        Data generator for 2D model training
        Parameters
        ----------
        img : [[str, int]]
            path to volume followed by volume slice index
        label : [[str, int]]
            path to volume followed by volume slice index
        augmentation : albumentations.Compose
            composition of transformations
        size : int
            batch size
        shuffle : bool
            whether we should shuffle after each epoch
        weight : str, None
            apply sample weight for unbalanced dataset
            -"balanced" : (1/instances) * (total/nb_classes)
        """
        self.x = img
        self.y = label
        self.size = size
        self.aug = augmentation
        self.shuffle = shuffle
        if not weight is None:
            self.w = self.balanced_class_weights()
        if len(self.x) > 0:
            self.__missing_color_ch = len(tif.imread(self.x[0][0]).shape) < 4
    
    def __len__(self):
        # number of batch (some img might not be included in the epoch)
        return len(self.y) // self.size
        
    def __getitem__(self, idx):
        data_i = idx * self.size
        
        # instead of reading entire volume, reads only 1 slice using TiffFile
        x_info = tif.TiffFile(self.x[data_i][0])
        y_info = tif.TiffFile(self.y[data_i][0])

        x_batch = np.zeros((self.size, * x_info.pages[0].shape))
        y_batch = np.zeros((self.size, * y_info.pages[0].shape))
        
        # fill batches
        for i in range(self.size):
            x_path, slc = self.x[data_i]
            y_path, __  = self.y[data_i]
            x_info = tif.TiffFile(x_path)
            y_info = tif.TiffFile(y_path)
            data = {
                'image': x_info.pages[slc].asarray(),
                'mask' : y_info.pages[slc].asarray()
            }
            
            if not self.aug is None:
                data = self.aug(**data)
            
            x_batch[i] = data['image']
            y_batch[i] = data['mask']
            
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

        return batch
        
    def on_epoch_end(self):
        if self.shuffle:
            self.x, self.y = shuffle(self.x, self.y)

    def balanced_class_weights(self):
        """
        TODO make another python file for sample weight creation
        """
        # assume y is categorical (1 channel per class)
        weights = []
        
        list_of_volumes = list(set([name for name, slice in self.y]))
        if len(list_of_volumes) > 0:
            big_pile = tif.imread(list_of_volumes[0])
            for mask in list_of_volumes[1:]:
                big_pile += tif.imread(mask)

            nb_classes = big_pile.shape[-1]
            for i in range(nb_classes):
                total = big_pile.sum()
                class_instance = big_pile[..., i].sum()
                w_result = (1 / class_instance) * (total / nb_classes)
                weights.append(w_result)
            
        return weights