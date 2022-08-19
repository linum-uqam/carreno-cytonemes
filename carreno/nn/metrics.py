# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy

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


# TODO try out these metrics
# https://github.com/frankkramer-lab/MIScnn/blob/master/miscnn/neural_network/metrics.py
# asymmetric_focal_loss : for unbalanced dataset, better than Dice?