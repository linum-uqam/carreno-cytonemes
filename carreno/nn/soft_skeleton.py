#
# Code adapted from https://github.com/jocpae/clDice/blob/master/cldice_loss/keras/soft_skeleton.py
#

from keras import layers as KL
from keras import backend as K
import numpy as np
import tensorflow as tf

def soft_erode2D(img):
    """[This function performs soft-erosion operation on a float32 image]
    Args:
        img ([float32]): [image to be soft eroded]
    Returns:
        [float32]: [the eroded image]
    """
    return -KL.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', data_format=None)(-img)


def soft_erode3D(img):
    """[This function performs soft-erosion operation on a float32 image]
    Args:
        img ([float32]): [image to be soft eroded]
    Returns:
        [float32]: [the eroded image]
    """
    # changed to 3x3x3 to be compatible with dilation
    return -KL.MaxPool3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(-img)


def soft_dilate2D(img):
    """[This function performs soft-dilation operation on a float32 image]
    Args:
        img ([float32]): [image to be soft dialated]
    Returns:
        [float32]: [the dialated image]
    """
    return KL.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', data_format=None)(img)


def soft_dilate3D(img):
    """[This function performs soft-dilation operation on a float32 image]
    Args:
        img ([float32]): [image to be soft dialated]
    Returns:
        [float32]: [the dialated image]
    """
    return KL.MaxPool3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(img)


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