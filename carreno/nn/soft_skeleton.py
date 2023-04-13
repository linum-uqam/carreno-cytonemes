#
# Code adapted from https://github.com/jocpae/clDice/blob/master/cldice_loss/keras/soft_skeleton.py
#

import tensorflow as tf
from keras import layers as KL

def_mode = 0


def soft_dilate(ndim, img, mode=def_mode):
    """
    Soft-dilate operation on a float32 image
    Parameters
    ----------
    ndim : int
        2 for 2D, else 3D
    img : tf.tensor([float32])
        Image to be soft dilated
    mode : int
        Padding type
        - 0 : same
        - 1 : reflect
        - 2 : constant with 0
        
        Theorically, 'same' padding should dilate img padding into tensor with value 0
        (according to convolution layers doc), hence reflect, but it doesn't seem to
        change anything. Using our own constant padding of 0, the theory is proven right.
    Returns
    -------
    img : tf.tensor([float32])
        Dilated image
    """
    pool_fn = KL.MaxPool2D if ndim == 2 else KL.MaxPool3D
    pool_size = [3] * ndim
    strides   = [1] * ndim
    paddings = [[0,0], *([[1,1]] * ndim), [0,0]]
    
    if mode == 0:
        img = pool_fn(pool_size=pool_size, strides=strides, padding='same')(img)
    elif mode == 1:
        pad = tf.pad(img, paddings=paddings, mode='REFLECT')
        img = pool_fn(pool_size=pool_size, strides=strides, padding='valid')(pad)
    elif mode == 2:
        pad = tf.pad(img, paddings=paddings, mode='CONSTANT', constant_values=0)
        img = pool_fn(pool_size=pool_size, strides=strides, padding='valid')(pad)
    else:
        raise NotImplementedError
    
    return img


def soft_dilate2D(*args, **kwargs):
    return soft_dilate(2, *args, **kwargs)


def soft_dilate3D(*args, **kwargs):
    return soft_dilate(3, *args, **kwargs)


def soft_erode(dilate_fn, img, mode=def_mode):
    """
    Soft-erode operation on a float32 image
    Parameters
    ----------
    img : tf.tensor([float32])
        Image to be soft eroded
    mode : int
        Refer to soft_dilate2D
    Returns
    -------
    img : tf.tensor([float32])
        Eroded image
    """
    return -dilate_fn(-img, mode=mode)


def soft_erode2D(*args, **kwargs):
    return soft_erode(soft_dilate2D, *args, **kwargs)


def soft_erode3D(*args, **kwargs):
    return soft_erode(soft_dilate3D, *args, **kwargs)


def soft_open(erode_fn, dilate_fn, img, mode=def_mode):
    """
    Soft-open operation on a float32 image
    Parameters
    ----------
    img : tf.tensor([float32])
        Image to be soft opened
    mode : int
        Refer to soft_dilate2D
    Returns
    -------
    img : tf.tensor([float32])
        Opened image
    """
    img = erode_fn(img, mode=mode)
    img = dilate_fn(img, mode=mode)
    return img


def soft_open2D(*args, **kwargs):
    return soft_open(soft_erode2D, soft_dilate2D, *args, **kwargs)


def soft_open3D(*args, **kwargs):
    return soft_open(soft_erode3D, soft_dilate3D, *args, **kwargs)


def soft_skel(erode_fn, open_fn, img, iters=10, mode=def_mode):
    """
    Soft-skeleton operation on a float32 image
    Parameters
    ----------
    erode_fn : function
        erosion function
    open_fn : function
        opening function
    img : tf.tensor([float32])
        Image to be soft Skeletonized
    iters : int
        Number of thinning iterations
    mode : int
        Refer to soft_dilate2D
    Returns
    -------
    img : tf.tensor([float32])
        Skeletonized image
    """
    assert iters >= 0
    
    img1 = open_fn(img, mode=mode)
    skel = tf.nn.relu(img - img1)
    
    for i in range(iters):
        img = erode_fn(img, mode=mode)
        img1 = open_fn(img, mode=mode)
        delta = tf.nn.relu(img - img1)
        intersect = tf.math.multiply(skel, delta)
        skel += tf.nn.relu(delta - intersect)
    
    return skel


def soft_skel2D(*args, **kwargs):
    return soft_skel(soft_erode2D, soft_open2D, *args, **kwargs)


def soft_skel3D(*args, **kwargs):
    return soft_skel(soft_erode3D, soft_open3D, *args, **kwargs)


if __name__ == '__main__':
    import unittest
    import numpy as np

    class TestTransforms(unittest.TestCase):
        dot = np.zeros((5,5,5,2))
        dot[..., 0] = 1
        dot[2, 2, 2] = [0, 1]
        tensor = tf.expand_dims(tf.convert_to_tensor(dot, dtype=tf.float32), axis=0)
        half = dot.shape[0] // 2

        def test_erode(self):
            x = self.__class__.tensor
            h = self.__class__.half
            expected = np.zeros((5,5,5,2))
            expected[..., 0] = 1
            expected[1:-1, 1:-1, 1:-1] = 0
            y = tf.expand_dims(tf.convert_to_tensor(expected, dtype=tf.float32), 0)
            
            # same pad
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_erode2D(img=x[:, h], mode=0), y[:, h])))
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_erode3D(img=x      , mode=0), y)))

            # reflect pad
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_erode2D(img=x[:, h], mode=1), y[:, h])))
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_erode3D(img=x      , mode=1), y)))

        def test_dilate(self):
            x = self.__class__.tensor
            h = self.__class__.half
            expected = np.zeros((5,5,5,2))
            expected[..., 0] = 1
            expected[1:-1, 1:-1, 1:-1, 1] = 1
            y = tf.expand_dims(tf.convert_to_tensor(expected, dtype=tf.float32), 0)
            
            # same pad
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_dilate2D(img=x[:, h], mode=0), y[:, h])))
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_dilate3D(img=x      , mode=0), y)))

            # reflect pad
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_dilate2D(img=x[:, h], mode=1), y[:, h])))
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_dilate3D(img=x      , mode=1), y)))
        
        def test_open(self):
            x = self.__class__.tensor
            h = self.__class__.half
            expected = np.zeros((5,5,5,2))
            expected[..., 0] = 1
            expected[2, 2, 2] = 0
            y = tf.expand_dims(tf.convert_to_tensor(expected, dtype=tf.float32), 0)
            
            # same pad
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_open2D(img=x[:, h], mode=0), y[:, h])))
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_open3D(img=x      , mode=0), y)))
            
            # reflect pad
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_open2D(img=x[:, h], mode=1), y[:, h])))
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_open3D(img=x      , mode=1), y)))
        
        def test_skel(self):
            x = self.__class__.tensor
            h = self.__class__.half
            expected = np.zeros((5,5,5,2))
            expected[2, 2, 2] = [0, 1]
            y = tf.expand_dims(tf.convert_to_tensor(expected, dtype=tf.float32), 0)

            # same pad
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_skel2D(img=x[:, h], iters=0, mode=0), y[:, h])))
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_skel3D(img=x,       iters=0, mode=0), y)))

            # reflect pad
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_skel2D(img=x[:, h], iters=0, mode=1), y[:, h])))
            self.assertTrue(tf.reduce_all(tf.math.equal(soft_skel3D(img=x      , iters=0, mode=1), y)))
            
    unittest.main()

    # Visualization of clDice inner working
    # (comment out `unittest.main()` to run)
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.filters import gaussian

    cube = np.zeros((20, 20, 20, 3))
    cube[4:-4, 4:, 4:-4] = [0, 1, 0]
    cube[:4]        = [1, 0, 0]
    cube[-4:]       = [1, 0, 0]
    cube[:, :4]     = [1, 0, 0]
    cube[:, :, :4]  = [1, 0, 0]
    cube[:, :, -4:] = [1, 0, 0]
    cube = gaussian(cube, sigma=1.05, multichannel=True)
    cube_tensor = tf.expand_dims(tf.convert_to_tensor(cube, dtype=tf.float32), 0)

    n = cube.shape[0] // 2

    def __plt_skel(img, iters, n, mode=0):
        img1 = soft_open3D(img, mode)
        skel = tf.nn.relu(img-img1)

        for j in range(iters):
            plt.subplot(231)
            plt.title('img')
            plt.imshow(img.numpy()[0,n])

            plt.subplot(232)
            plt.title('erosion')
            img = soft_erode3D(img, mode)
            plt.imshow(img.numpy()[0,n])

            plt.subplot(233)
            plt.title('opening')
            img1 = soft_open3D(img, mode)
            plt.imshow(img1.numpy()[0,n])

            plt.subplot(234)
            plt.title('delta')
            delta = tf.nn.relu(img-img1)
            plt.imshow(delta.numpy()[0,n])

            plt.subplot(235)
            plt.title('intersection')
            intersect = tf.math.multiply(skel, delta)
            plt.imshow(intersect.numpy()[0,n])

            plt.subplot(236)
            plt.title('skeleton')
            skel += tf.nn.relu(delta-intersect)
            plt.imshow(skel.numpy()[0,n])
            
            plt.show()

    __plt_skel(cube_tensor, 7, n, mode=2)
    
    