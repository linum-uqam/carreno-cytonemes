# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as K
from keras import layers as KL
from tensorflow.keras.losses import CategoricalCrossentropy
import scipy
import numpy as np
from carreno.nn.soft_skeleton import soft_skel2D, soft_skel3D

#-------------#
# Coefficient #
#-------------#

def dice_score(smooth=1.):
    """
    Gets the function to calculate the Dice score between 2 TensorFlow tensors.
    Parameters
    ----------
    smooth : float
        Added to numerator and denominator to avoid division by 0
    Returns
    -------
    metric_function : function
        Function to calculate Dice score between target and prediction
    """
    def metric_function(y_true, y_pred):
        """
        Dice score between 0 (worst) and 1 (best).
        Parameters
        ----------
        y_true : tf.Tensor
            Target tensor
        y_pred : tf.Tensor
            Prediction tensor
        Returns
        -------
        score : float
            Dice score
        """
        y_true_f = tf.reshape(y_true, [-1])  # flatten
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        numetator = (2. * intersection + smooth)
        denominator = (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        score = numetator / denominator
        return score
    
    # for metric name output during tensorflow fitting
    metric_function.__name__ = "dice"

    return metric_function


#--------#
# Losses #
#--------#


def dice_loss(y_true, y_pred):
    """
    Dice loss between 0 (best) and 1 (worst). Best used for unbalanced dataset over IoU
    Parameters
    ----------
    y_true : tf.Tensor
        Target tensor
    y_pred : tf.Tensor
        Prediction tensor
    Returns
    -------
    loss : float
        Loss score
    """
    dice = dice_score(smooth=1.)
    loss = 1 - dice(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    """
    Dice loss additionned to categorical cross entropy for possibly simplifying convergence.
    Taken from https://notebook.community/cshallue/models/samples/outreach/blogs/segmentation_blogpost/image_segmentation
    Parameters
    ----------
    y_true : tf.Tensor
        Target tensor
    y_pred : tf.Tensor
        Prediction tensor
    Returns
    -------
    loss : float
        Loss score
    """
    cce = CategoricalCrossentropy()
    loss = cce(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


focal_loss = tfa.losses.SigmoidFocalCrossEntropy(gamma=2., alpha=.25)


def focal_bce_dice_loss(y_true, y_pred):
    # https://arxiv.org/ftp/arxiv/papers/2209/2209.00729.pdf
    return bce_dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)


def dice_cldice2D_loss(iters=10, alpha=0.5):
    """[function to compute dice+cldice loss]
    Args:
        iters (int, optional): [skeletonization iteration]. Defaults to 10.
        alpha (float, optional): [weight for the cldice component]. Defaults to 0.5.
    """
    def loss(y_true, y_pred):
        """[summary]
        Args:
            y_true ([float32]): [ground truth image]
            y_pred ([float32]): [predicted image]
        Returns:
            [float32]: [loss value]
        """
        smooth = 1.
        skel_pred = soft_skel2D(y_pred, iters)
        skel_true = soft_skel2D(y_true, iters)
        pres = (K.sum(tf.math.multiply(skel_pred, y_true))+smooth)/(K.sum(skel_pred)+smooth)    
        rec  = (K.sum(tf.math.multiply(skel_true, y_pred))+smooth)/(K.sum(skel_true)+smooth)    
        cl_dice = 1.- 2.0*(pres*rec)/(pres+rec)
        dice = dice_loss(y_true, y_pred)
        return (1.0-alpha)*dice+alpha*cl_dice
    return loss


def dice_cldice3D_loss(iters=10, alpha=0.5):
    """[function to compute dice+cldice loss]
    Args:
        iters (int, optional): [skeletonization iteration]. Defaults to 10.
        alpha (float, optional): [weight for the cldice component]. Defaults to 0.5.
    """
    def loss(y_true, y_pred):
        """[summary]
        Args:
            y_true ([float32]): [ground truth image]
            y_pred ([float32]): [predicted image]
        Returns:
            [float32]: [loss value]
        """
        smooth = 1.
        skel_pred = soft_skel3D(y_pred, iters)
        skel_true = soft_skel3D(y_true, iters)
        pres = (K.sum(tf.math.multiply(skel_pred, y_true))+smooth)/(K.sum(skel_pred)+smooth)    
        rec  = (K.sum(tf.math.multiply(skel_true, y_pred))+smooth)/(K.sum(skel_true)+smooth)    
        cl_dice = 1.- 2.0*(pres*rec)/(pres+rec)
        dice = dice_loss(y_true, y_pred)
        return (1.0-alpha)*dice+alpha*cl_dice
    return loss

"""
def dice_cldice3D_loss(iters=10, alpha=0.5):
    def loss(y_true, y_pred):
        smooth = 1.
        skel_pred = soft_skel3D(y_pred, iters)
        skel_true = soft_skel3D(y_true, iters)
        pres = (K.sum(tf.math.multiply(skel_pred, y_true))+smooth)/(K.sum(skel_pred[:,1:,:,:,:])+smooth)    
        rec = (K.sum(tf.math.multiply(skel_true, y_pred)[:,1:,:,:,:])+smooth)/(K.sum(skel_true[:,1:,:,:,:])+smooth)    
        cl_dice = 1.- 2.0*(pres*rec)/(pres+rec)
        dice = dice_loss(y_true, y_pred)
        return (1.0-alpha)*dice+alpha*cl_dice
    return loss
"""


def adap_wing_loss(theta=0.5, alpha=2.1, omega=14, epsilon=1):
    """
    Adaptive Wing loss. Used for heatmap ground truth.
    This code is the Keras adaptation of ivadomed implementation of AdapWingLoss with PyTorch
    # https://github.com/ivadomed/ivadomed/blob/bd904e5e139fa6a437abfc225216f6057824f1e3/ivadomed/losses.py
    Parameters
    ----------
    theta : float
        Threshold between linear and non linear loss.
    alpha : float
        Used to adapt loss shape to input shape and make loss smooth at 0 (background).
        It needs to be slightly above 2 to maintain ideal properties.
    omega : float
        Multiplicating factor for non linear part of the loss.
    epsilon : float
        factor to avoid gradient explosion. It must not be too small
    Returns
    -------
    metric_function : function
        Function to calculate Dice score between target and prediction
    """

    def loss_function(input, target):
        eps = epsilon
        # Compute adaptative factor
        A = omega * (1 / (1 + tf.pow(theta / eps,
                                             alpha - target))) * \
            (alpha - target) * tf.pow(theta / eps,
                                              alpha - target - 1) * (1 / eps)

        # Constant term to link linear and non linear part
        C = (theta * A - omega * tf.math.log(1 + tf.pow(theta / eps, alpha - target)))

        batch_size = target.shape[0]
        
        mask = np.zeros_like(target)
        kernel = scipy.ndimage.generate_binary_structure(2, 2)
        
        # For 3D segmentation tasks
        if len(input.shape) == 5:
            kernel = scipy.ndimage.generate_binary_structure(3, 2)

        for i in range(batch_size if batch_size else 0):
            img_list = list()
            img_list.append(np.round(target[i].cpu().numpy() * 255))
            img_merge = np.concatenate(img_list)
            img_dilate = scipy.ndimage.binary_opening(img_merge, np.expand_dims(kernel, axis=0))
            img_dilate[img_dilate < 51] = 1  # 0*omega+1
            img_dilate[img_dilate >= 51] = 1 + omega  # 1*omega+1
            img_dilate = np.array(img_dilate, dtype=int)

            mask[i] = img_dilate

        diff_hm = tf.abs(target - input)
        AWingLoss = A * diff_hm - C
        tmp = AWingLoss.numpy()
        idx = diff_hm < theta
        tmp[idx] = omega * tf.math.log(1 + tf.pow(diff_hm / eps, alpha - target)).numpy()[idx]
        AWingLoss = tf.convert_to_tensor(tmp)

        AWingLoss *= tf.constant(mask)
        sum_loss = tf.math.reduce_sum(AWingLoss)
        #all_pixel = tf.sum(mask)
        mean_loss = sum_loss  # / all_pixel

        return mean_loss
    
    # for metric name output during tensorflow fitting
    loss_function.__name__ = "AdaWingLoss"
    
    return loss_function
