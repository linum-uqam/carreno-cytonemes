# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class Generator(tf.keras.utils.Sequence):
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
        augmentation : carreno.processing.transforms.Transform or None
            Transformation(s) to apply
        shuffle : bool
            whether we should shuffle after each epoch
        """
        assert len(vol) >= size, "Batch size is bigger then number of data instances, {} > {}".format(size, len(vol))
        self.x = vol
        self.y = label
        self.w = weight
        self.size = size
        self.aug = tfs.Compose([]) if augmentation is None else augmentation
        self.shuffle = shuffle
    
    def __len__(self):
        # number of batch (aka steps) to go through all entries
        return len(self.y) // self.size
        
    def __getitem__(self, idx):
        """
        Get batch of inputs targets and weights for batch #idx.
        Parameters
        ----------
        idx : int
            Batch id, range from 0 to self.__len__()
        Returns
        -------
        batch : tuple([x], [y], [w])
            Batch content, each categories seperated in their own list
        """
        data_i = idx * self.size
        
        # instead of reading entire volume, reads only 1 slice using TiffFile
        x_batch = []
        y_batch = []
        w_batch = []

        # fill batches
        for i in range(self.size):
            data = self.aug(self.x[data_i],
                            self.y[data_i],
                            None if self.w is None else self.w[data_i])
            
            # fill batch w augmentation
            x_batch.append(data[0])
            y_batch.append(data[1])
            if self.w:
                w_batch.append(data[2])
            
            data_i += 1

        # convert list of data to tensor
        x_batch = tf.convert_to_tensor(x_batch)
        y_batch = tf.convert_to_tensor(y_batch)
        w_batch = tf.convert_to_tensor(w_batch)

        batch = [x_batch, y_batch, w_batch][:2+(not self.w is None)]

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

    class TestGenerator(unittest.TestCase):
        x, y, w = np.meshgrid(range(5), range(5), range(5))
        y = np.stack([y]*3, axis=-1)

        def test_init(self):
            a, b = self.__class__.x, self.__class__.y
            n = 2

            # batch size small enough for data
            gen = Generator([a]*n, [b]*n, None, size=n, augmentation=None)

            batch = next(iter(gen))

            self.assertEqual(2, len(batch))
            self.assertEqual(2, len(batch[0]))
            self.assertTrue((a == batch[0][0].numpy()).all())
            self.assertTrue((b == batch[1][0].numpy()).all())
            
            # batch size too big for data
            self.assertRaises(AssertionError, Generator, [a], [b], None, n, None)

        def test_2d(self):
            a, b, c = self.__class__.x, self.__class__.y, self.__class__.w

            aug2D = tfs.Compose([
                tfs.Sample([1]),
                tfs.Squeeze(axis=0),
                tfs.Stack(axis=-1, n=1)
            ])
            gen2D = Generator([a], [b], [c], size=1, augmentation=aug2D)

            batch = next(iter(gen2D))
            
            self.assertEqual(3, len(batch))
            self.assertEqual(1, len(batch[0]))
            self.assertTrue((np.expand_dims(a, axis=-1) == np.expand_dims(batch[0][0].numpy(), axis=0)).all(axis=1).all(axis=1).any())
            self.assertTrue((b == np.expand_dims(batch[1][0].numpy(), axis=0)).all(axis=1).all(axis=1).any())
            self.assertTrue((c == np.expand_dims(batch[2][0].numpy(), axis=0)).all(axis=1).all(axis=1).any())

        def test_3d(self):
            a, b, c = self.__class__.x, self.__class__.y, self.__class__.w
            n = 3
            aug3D = tfs.Compose([
                tfs.Sample([5,5,5]),
                tfs.Stack(axis=-1, n=n)
            ])
            gen3D = Generator([a], [b], [c], size=1, augmentation=aug3D)

            batch = next(iter(gen3D))
            
            self.assertEqual(3, len(batch))
            self.assertEqual(1, len(batch[0]))
            self.assertTrue((np.expand_dims([a]*n, axis=-1) == batch[0][0].numpy()).all())
            self.assertTrue((b == batch[1][0].numpy()).all())
            self.assertTrue((c == batch[2][0].numpy()).all())
    
    unittest.main()