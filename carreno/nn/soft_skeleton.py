#
# Code adapted from https://github.com/jocpae/clDice/blob/master/cldice_loss/keras/soft_skeleton.py
#

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import backend as K


def soft_erode2D(img):
    """[This function performs soft-erosion operation on a float32 image]
    Args:
        img ([float32]): [image to be soft eroded]
    Returns:
        [float32]: [the eroded image]
    """
    return -KL.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(-img)


def soft_erode3D(img):
    """[This function performs soft-erosion operation on a float32 image]
    Args:
        img ([float32]): [image to be soft eroded]
    Returns:
        [float32]: [the eroded image]
    """
    # changed to 3x3x3 to be compatible with dilation
    return -KL.MaxPool3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(-img)


def soft_dilate2D(img):
    """[This function performs soft-dilation operation on a float32 image]
    Args:
        img ([float32]): [image to be soft dialated]
    Returns:
        [float32]: [the dialated image]
    """
    return KL.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(img)


def soft_dilate3D(img):
    """[This function performs soft-dilation operation on a float32 image]
    Args:
        img ([float32]): [image to be soft dialated]
    Returns:
        [float32]: [the dialated image]
    """
    return KL.MaxPool3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(img)


def soft_open2D(img):
    """[This function performs soft-open operation on a float32 image]
    Args:
        img ([float32]): [image to be soft opened]
    Returns:
        [float32]: [image after soft-open]
    """
    img = soft_erode2D(img)
    img = soft_dilate2D(img)
    return img


def soft_open3D(img):
    """[This function performs soft-open operation on a float32 image]
    Args:
        img ([float32]): [image to be soft opened]
    Returns:
        [float32]: [image after soft-open]
    """
    img = soft_erode3D(img)
    img = soft_dilate3D(img)
    return img


def soft_skel2D(img, iters):
    """[summary]
    Args:
        img ([float32]): [description]
        iters ([int]): [description]
    Returns:
        [float32]: [description]
    """
    img1 = soft_open2D(img)
    skel = tf.nn.relu(img-img1)

    for j in range(iters):
        img = soft_erode2D(img)
        img1 = soft_open2D(img)
        delta = tf.nn.relu(img-img1)
        intersect = tf.math.multiply(skel, delta)
        skel += tf.nn.relu(delta-intersect)
    return skel


def soft_skel3D(img, iters):
    """[summary]
    Args:
        img ([float32]): [description]
        iters ([int]): [description]
    Returns:
        [float32]: [description]
    """
    img1 = soft_open3D(img)
    skel = tf.nn.relu(img-img1)

    for j in range(iters):
        img = soft_erode3D(img)
        img1 = soft_open3D(img)
        delta = tf.nn.relu(img-img1)
        intersect = tf.math.multiply(skel, delta)
        skel += tf.nn.relu(delta-intersect)
    return skel


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    cube = np.zeros((25, 25, 25, 3))
    cube[5:-5, 5:-5, 5:-5] = [0, 1, 0]
    cube[:5]        = [1, 0, 0]
    cube[-5:]       = [1, 0, 0]
    cube[:, :5]     = [1, 0, 0]
    cube[:, -5:]    = [1, 0, 0]
    cube[:, :, :5]  = [1, 0, 0]
    cube[:, :, -5:] = [1, 0, 0]
    cube_tensor = tf.expand_dims(tf.convert_to_tensor(cube, dtype=tf.float32), 0)

    n = 12

    def __plt_skel(img, iters, n):
        img1 = soft_open3D(img)
        skel = tf.nn.relu(img-img1)

        for j in range(iters):
            plt.subplot(151)
            plt.title('erosion')
            img = soft_erode3D(img)
            plt.imshow(img.numpy()[0,n])

            plt.subplot(152)
            plt.title('opening')
            img1 = soft_open3D(img)
            plt.imshow(img1.numpy()[0,n])

            plt.subplot(153)
            plt.title('delta')
            delta = tf.nn.relu(img-img1)
            plt.imshow(delta.numpy()[0,n])

            plt.subplot(154)
            plt.title('intersection')
            intersect = tf.math.multiply(skel, delta)
            plt.imshow(intersect.numpy()[0,n])

            plt.subplot(155)
            plt.title('skeleton')
            skel += tf.nn.relu(delta-intersect)
            plt.imshow(skel.numpy()[0,n])
            
            plt.show()

    __plt_skel(cube_tensor, 15, n)
