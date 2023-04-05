# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class VolumeGenerator(tf.keras.utils.Sequence):
    def __init__(self, vol, label, weight=None, size=1, augmentation=None, shuffle=True):
        """
        Data generator.
        Parameters
        ----------
        vol : list
            Input data
        label : list
            Target data
        weight : list
            Weight data
        size : int
            batch size
        augmentation : carreno.processing.transforms.Transform
            Transformation(s) to apply
        shuffle : bool
            whether we should shuffle after each epoch
        """
        assert len(vol) >= size, "Batch size is bigger then number of data instances, {} > {}".format(size, len(vol))
        self.x = vol
        self.y = label
        self.w = weight
        self.size = size
        self.aug = augmentation
        self.shuffle = shuffle
    
    def __len__(self):
        # number of batch (some img might not be included in the epoch)
        return len(self.y) // self.size
        
    def __getitem__(self, idx):
        data_i = idx * self.size
        
        # instead of reading entire volume, reads only 1 slice using TiffFile
        batch = []

        # fill batches
        for i in range(self.size):
            data = self.aug(self.x[data_i],
                            self.y[data_i],
                            None if self.w is None else self.w[data_i])
            
            # fill batch w augmentation
            subject = [data[0], data[1]]
            if self.w:
                subject.append(data[2])
            batch.append(subject)
            data_i += 1

        return batch
        
    def on_epoch_end(self):
        if self.shuffle:
            if self.w is None:
                self.x, self.y = shuffle(self.x, self.y)
            else:
                self.x, self.y, self.w = shuffle(self.x, self.y, self.w)

if __name__ == '__main__':
    import unittest
    import carreno.processing.transforms as tfs

    class TestStringMethods(unittest.TestCase):
        x, y, w = np.meshgrid(range(5), range(5), range(5))
        
        def test_init(self):
            a, b = self.__class__.x, self.__class__.y
            aug = tfs.Compose([])
            n = 2
            gen = VolumeGenerator([a]*n, [b]*n, None, size=n, augmentation=aug)

            batch = next(iter(gen))
            data = batch[0]

            self.assertEqual(2, len(batch))
            self.assertEqual(2, len(data))
            self.assertTrue((a == data[0]).all())
            self.assertTrue((b == data[1]).all())
        
        def test_2d(self):
            a, b, c = self.__class__.x, self.__class__.y, self.__class__.w

            aug2D = tfs.Compose([
                tfs.Sample([1]),
                tfs.Squeeze(axis=0),
                tfs.Stack(axis=-1, n=1)
            ])
            gen2D = VolumeGenerator([a], [b], [c], size=1, augmentation=aug2D)

            batch = next(iter(gen2D))
            data = batch[0]

            self.assertEqual(1, len(batch))
            self.assertEqual(3, len(data))
            self.assertTrue((np.expand_dims(a, axis=-1) == np.expand_dims(data[0], axis=0)).all(axis=1).all(axis=1).any())
            self.assertTrue((b == np.expand_dims(data[1], axis=0)).all(axis=1).all(axis=1).any())
            self.assertTrue((c == np.expand_dims(data[2], axis=0)).all(axis=1).all(axis=1).any())

        def test_3d(self):
            a, b, c = self.__class__.x, self.__class__.y, self.__class__.w
            n = 3
            aug3D = tfs.Compose([
                tfs.Sample([5,5,5]),
                tfs.Stack(axis=-1, n=n)
            ])
            gen3D = VolumeGenerator([a], [b], [c], size=1, augmentation=aug3D)

            batch = next(iter(gen3D))
            data = batch[0]

            self.assertEqual(1, len(batch))
            self.assertEqual(3, len(data))
            self.assertTrue((np.expand_dims([a]*n, axis=-1) == data[0]).all())
            self.assertTrue((b == data[1]).all())
            self.assertTrue((c == data[2]).all())
    
    unittest.main()