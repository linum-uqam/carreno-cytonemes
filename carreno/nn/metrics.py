# -*- coding: utf-8 -*-
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.losses import CategoricalCrossentropy
from carreno.nn.soft_skeleton import soft_skel2D, soft_skel3D
import logging

logging.basicConfig(filename='example.log', level=logging.DEBUG)

class Metric():
    def __init__(self, class_weight=None):
        self.class_weight = None if class_weight is None else class_weight / np.mean(class_weight)

    def coefficient(self, y_true, y_pred, sample_weight=None):
        raise NotImplementedError
    
    coefficient.__name__ = "coef"
    
    def loss(self, y_true, y_pred, sample_weight=None):
        raise NotImplementedError
    
    loss.__name__ = "loss"
    

class Dice(Metric):
    def __init__(self, smooth=1., class_weight=None):
        """
        Compute Dice coefficient and loss.
        Best used for unbalanced dataset over IoU.
        Parameters
        ----------
        smooth : float
            Smoothing factor added to numerator and denominator when dividing
        class_weight : [float]
            List of weights per classes. Not supported with sample weights
        """
        super().__init__(class_weight)
        self.smooth = tf.cast(smooth, dtype=tf.float32)
    
    def coefficient(self, y_true, y_pred, sample_weight=None):
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
        score = 1

        if sample_weight is None and self.class_weight is None:
            y_true_f = tf.reshape(y_true, [-1])  # flatten
            y_pred_f = tf.reshape(y_pred, [-1])
            intersection = tf.cast(tf.reduce_sum(y_true_f * y_pred_f), dtype=tf.float32)
            numetator = 2. * intersection + self.smooth
            denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth
            score = numetator / denominator
        elif not sample_weight is None:
            intersection = y_true * y_pred
            sample_smooth = self.smooth / tf.size(y_true.shape, out_type=tf.float32)
            numetator = 2. * intersection + self.smooth + sample_smooth
            denominator = y_true + y_pred + sample_smooth
            score_wo_weight = numetator / denominator
            score = tf.reduce_mean(score_wo_weight * tf.expand_dims(sample_weight, axis=-1))
        elif not self.class_weight is None:
            y_true_f = tf.reshape(y_true, [-1, len(self.class_weight)])  # flatten each classes
            y_pred_f = tf.reshape(y_pred, [-1, len(self.class_weight)])
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            numetator = 2. * intersection + self.smooth
            denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth
            score_wo_weight = numetator / denominator
            score = tf.reduce_mean(score_wo_weight * self.class_weight)
        else:
            raise NotImplementedError
        
        return score

    coefficient.__name__ = "dice"
    
    def loss(self, y_true, y_pred, sample_weight=None):
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
        return 1 - self.coefficient(y_true, y_pred, sample_weight)
    
    loss.__name__ = "dice_loss"


class CeDice(Metric):
    def __init__(self, smooth=1., class_weight=None):
        """
        Compute Dice with binary cross-entropy loss.
        Vaguely used for segmentation, but can't find it's origin. Used by nnUNet in 2018 as SOTA.
        Parameters
        ----------
        smooth : float
            Smoothing factor added to numerator and denominator when dividing
        class_weight : [float]
            List of weights per classes. Not supported with sample weights
        """
        super().__init__(class_weight)
        self.dice = Dice(smooth=smooth, class_weight=self.class_weight)
        self.ce = CategoricalCrossentropy()

    def loss(self, y_true, y_pred, sample_weight=None):
        """
        CE Dice loss between 0 (best) and infitity (worst).
        Parameters
        ----------
        y_true : tf.Tensor
            Target tensor
        y_pred : tf.Tensor
            Prediction tensor
        Returns
        -------
        loss : float
            CE Dice loss
        """
        # cannot use dice loss function with `super` since coefficient function is overridden
        return self.dice.loss(y_true, y_pred, sample_weight) + self.ce(y_true, y_pred, sample_weight)

    loss.__name__ = "cedice_loss"


class ClDice(Metric):
    def __init__(self, iters=10, ndim=2, mode=2, cls=slice(0, None),
                 smooth=1., class_weight=None):
        """
        Compute clDice coefficient and loss.
        Parameters
        ----------
        iters : int
            Skeletonization iteration
        ndim : int
            Number of dimensions for data (not including feature channels)
        mode : int
            Padding type
            Refer to carreno.nn.soft_skeleton.soft_dilate2D
        cls : slice
            Slice of classes to consider for skeletonization
        smooth : float
            Smoothing factor added to numerator and denominator when dividing
        class_weight : [float]
            List of weights per classes. Not supported with sample weights
        """
        assert ndim > 1 and ndim < 4, "Incompatible dimensions, expected between 2 and 3, got {}".format(ndim)
        super().__init__(class_weight)
        self.iters = iters
        self.mode = mode
        self.cls = cls
        self.skel_fn = soft_skel2D if ndim == 2 else soft_skel3D
        self.smooth = tf.cast(smooth, dtype=tf.float32)
    
    def coefficient(self, y_true, y_pred, sample_weight=None):
        """
        clDice score between 0 (worst) and 1 (best).
        Parameters
        ----------
        y_true : tf.Tensor
            Target tensor
        y_pred : tf.Tensor
            Prediction tensor
        Returns
        -------
        score : float
            clDice score
        """
        y_pred_sample = tf.cast(y_pred[..., self.cls], dtype=tf.float32)
        y_true_sample = tf.cast(y_true[..., self.cls], dtype=tf.float32)
        skel_pred = self.skel_fn(img=y_pred_sample, iters=self.iters, mode=self.mode)
        skel_true = self.skel_fn(img=y_true_sample, iters=self.iters, mode=self.mode)
            
        if self.class_weight is None and sample_weight is None:
            prec = (K.sum(tf.math.multiply(skel_pred, y_true_sample)) + self.smooth) /\
                   (K.sum(skel_pred) + self.smooth)    
            rec  = (K.sum(tf.math.multiply(skel_true, y_pred_sample)) + self.smooth) /\
                   (K.sum(skel_true) + self.smooth)
            score = 2.0 * (prec * rec) / (prec + rec)
        elif not sample_weight is None:
            sample_smooth = self.smooth / tf.cast(tf.size(y_true.shape), dtype=tf.float32)
            prec_wo_weight = (tf.math.multiply(skel_pred, y_true_sample) + sample_smooth) /\
                             (skel_pred + sample_smooth)
            prec = prec_wo_weight * tf.expand_dims(sample_weight, axis=-1)
            rec_wo_weight = (tf.math.multiply(skel_true, y_pred_sample) + sample_smooth) /\
                            (skel_true + sample_smooth)
            rec = rec_wo_weight * tf.expand_dims(sample_weight, axis=-1)
            score = tf.reduce_mean(2.0 * (prec * rec) / (prec + rec))
        elif not self.class_weight is None:
            ndim_per_class = y_true.ndim - 1
            prec_wo_weight = (K.sum(tf.math.multiply(skel_pred, y_true_sample), axis=range(ndim_per_class)) + self.smooth) /\
                             (K.sum(skel_pred, axis=range(ndim_per_class)) + self.smooth)
            prec = prec_wo_weight * self.class_weight
            rec_wo_weight = (K.sum(tf.math.multiply(skel_true, y_pred_sample), axis=range(ndim_per_class)) + self.smooth) /\
                            (K.sum(skel_true, axis=range(ndim_per_class)) + self.smooth)
            rec = rec_wo_weight * self.class_weight
            score = tf.reduce_mean(2.0 * (prec * rec) / (prec + rec))
        else:
            raise NotImplementedError
            
        return score
    
    coefficient.__name__ = "cldice"

    def loss(self, y_true, y_pred, sample_weight=None):
        """
        clDice loss between 0 (best) and 1 (worst).
        Parameters
        ----------
        y_true : tf.Tensor
            Target tensor
        y_pred : tf.Tensor
            Prediction tensor
        Returns
        -------
        loss : float
            clDice loss
        """
        return 1. - self.coefficient(y_true, y_pred, sample_weight)

    loss.__name__ = "cldice_loss"


class DiceClDice(Metric):
    def __init__(self, alpha=0.5, iters=10, ndim=2, mode=2, cls=slice(0, None),
                 smooth=1, class_weight=None):
        """
        Compute Dice with clDice coefficient and loss
        Parameters
        ----------
        alpha : float
            Ratio for clDice proportion vs Dice
        iters : int
            Skeletonization iteration
        ndim : int
            Number of dimensions for data (not including feature channels)
        mode : int
            Padding type
            Refer to carreno.nn.soft_skeleton.soft_dilate2D
        cls : slice
            Class to consider for skeletonization
        smooth : float
            Smoothing factor added to numerator and denominator when dividing
        class_weight : [float]
            List of weights per classes. Not supported with sample weights
        """
        super().__init__(class_weight)
        self.alpha = alpha
        self.dice = Dice(smooth=smooth,
                         class_weight=self.class_weight)
        self.cldice = ClDice(iters=iters,
                             ndim=ndim,
                             cls=cls,
                             mode=mode,
                             smooth=smooth,
                             class_weight=self.class_weight)
    
    def coefficient(self, y_true, y_pred, sample_weight=None):
        """
        Dice + clDice score between 0 (worst) and 1 (best).
        Parameters
        ----------
        y_true : tf.Tensor
            Target tensor
        y_pred : tf.Tensor
            Prediction tensor
        Returns
        -------
        score : float
            Dice + clDice score
        """
        dice_value   = (1 - self.alpha) * self.dice.coefficient(y_true, y_pred, sample_weight)
        cldice_value = self.alpha     * self.cldice.coefficient(y_true, y_pred, sample_weight)
        return tf.cast(dice_value, dtype=tf.float32) + tf.cast(cldice_value, dtype=tf.float32)
    
    coefficient.__name__ = "dicecldice"

    def loss(self, y_true, y_pred, sample_weight=None):
        """
        Dice + clDice loss between 0 (best) and 1 (worst).
        Parameters
        ----------
        y_true : tf.Tensor
            Target tensor
        y_pred : tf.Tensor
            Prediction tensor
        Returns
        -------
        loss : float
            Dice + clDice loss
        """
        dice_value   = (1 - self.alpha) * self.dice.loss(y_true, y_pred, sample_weight)
        cldice_value = self.alpha     * self.cldice.loss(y_true, y_pred, sample_weight)
        return dice_value + cldice_value

    loss.__name__ = "dicecldice_loss"


class AdaptiveWingLoss(Metric):
    def __init__(self, alpha=float(2.1), omega=float(5), epsilon=float(1), theta=float(0.5),
                 class_weight=None):
        """
        Adaptive Wing loss. Used for heatmap ground truth.
        Suggested by SoftSeg article for non polarized region edges compared to Dice.

        Code based on https://github.com/SerdarHelli/TensorflowWorks/blob/main/Losses/Adaptive_Wing_Loss.py
        Parameters
        ----------
        alpha : float
            Used to adapt loss shape to input shape and make loss smooth at 0 (background).
            It needs to be slightly above 2 to maintain ideal properties.
        omega : float
            Multiplicating factor for non linear part of the loss.
        epsilon : float
            factor to avoid gradient explosion. It must not be too small
        theta : float
            Threshold between linear and non linear loss.
        class_weight : [float]
            List of weights per classes. Not supported with sample weights
        Returns
        -------
        metric_function : function
            Function to calculate Dice score between target and prediction
        """
        super().__init__(class_weight)
        self.alpha=alpha
        self.omega=omega
        self.epsilon=epsilon
        self.theta=theta

    def loss(self, y_true, y_pred, sample_weight=None):
        """
        Adaptive Wing loss between 0 (best) and infinity (worst).
        Parameters
        ----------
        y_true : tf.Tensor
            Target tensor
        y_pred : tf.Tensor
            Prediction tensor
        Returns
        -------
        loss : float
            Adaptive Wing loss
        """
        tde = self.theta / self.epsilon
        amt = self.alpha - y_true
        dif = tf.math.abs(y_true - y_pred)
        A = self.omega * (1 / (1 + tde ** amt)) * amt * (tde ** (amt - 1)) / self.epsilon
        C = self.theta * A - self.omega * tf.math.log(1 + tde ** amt)
        loss = tf.where(tf.math.greater_equal(dif, self.theta),
                        A * dif - C,
                        self.omega * tf.math.log(1 + dif / self.epsilon ** amt))
        if not self.class_weight is None:
            loss = loss * self.class_weight
        if not sample_weight is None:
            loss = loss * tf.expand_dims(sample_weight, axis=-1)
        return tf.reduce_mean(loss)

    loss.__name__ = "adawing_loss"


class ClDiceAdaptiveWingLoss(Metric):
    def __init__(self, iters=10, ndim=2, mode=2, cls=slice(0, None),
                 alpha=float(2.1), omega=float(5), epsilon=float(1), theta=float(0.5),
                 smooth=1, class_weight=None):
        """
        Adaptive Wing loss. Used for heatmap ground truth.
        Suggested by SoftSeg article for non polarized region edges compared to Dice.

        Code based on https://github.com/SerdarHelli/TensorflowWorks/blob/main/Losses/Adaptive_Wing_Loss.py
        Parameters
        ----------
        iters : int
            Skeletonization iteration
        ndim : int
            Number of dimensions for data (not including feature channels)
        mode : int
            Padding type
            Refer to carreno.nn.soft_skeleton.soft_dilate2D
        cls : slice
            Class to consider for skeletonization
        alpha : float
            Used to adapt loss shape to input shape and make loss smooth at 0 (background).
            It needs to be slightly above 2 to maintain ideal properties.
        omega : float
            Multiplicating factor for non linear part of the loss.
        epsilon : float
            factor to avoid gradient explosion. It must not be too small
        theta : float
            Threshold between linear and non linear loss.
        smooth : float
            Smoothing factor added to numerator and denominator when dividing
        class_weight : [float]
            List of weights per classes. Not supported with sample weights
        Returns
        -------
        metric_function : function
            Function to calculate Dice score between target and prediction
        """
        self.adawing = AdaptiveWingLoss(alpha=alpha,
                                        omega=omega,
                                        epsilon=epsilon,
                                        theta=theta,
                                        class_weight=class_weight)
        self.cldice = ClDice(iters=iters,
                             ndim=ndim,
                             cls=cls,
                             mode=mode,
                             smooth=smooth,
                             class_weight=class_weight)

    def loss(self, y_true, y_pred, sample_weight=None):
        """
        Adaptive Wing loss between 0 (best) and infinity (worst).
        Parameters
        ----------
        y_true : tf.Tensor
            Target tensor
        y_pred : tf.Tensor
            Prediction tensor
        Returns
        -------
        loss : float
            Adaptive Wing loss
        """
        return self.cldice.loss(y_true, y_pred, sample_weight) +\
               self.adawing.loss(y_true, y_pred, sample_weight)

    loss.__name__ = "adawingcldice_loss"


if __name__ == "__main__":
    import numpy as np
    import unittest

    class TestMetrics(unittest.TestCase):
        def test_dice(self):
            y = tf.convert_to_tensor([[0,1],[1,0]], dtype=tf.float32)
            p = tf.convert_to_tensor([[1,0],[0,1]], dtype=tf.float32)
            dice = Dice(smooth=1e-5)
            coef = dice.coefficient
            loss = dice.loss

            # coef
            self.assertEqual(0, round(coef(y, p).numpy(), 5))  # result != 0 because of smoothing
            self.assertEqual(1, coef(y, y).numpy())
            
            # loss
            self.assertEqual(1, round(loss(y, p).numpy(), 5))  # result != 0 because of smoothing
            self.assertEqual(0, loss(y, y).numpy())
    
        def test_CeDice(self):
            y = tf.convert_to_tensor([[0,1],[1,0]], dtype=tf.float32)
            p = tf.convert_to_tensor([[1,0],[0,1]], dtype=tf.float32)
            cedice = CeDice(smooth=1e-5)
            loss = cedice.loss

            # loss
            self.assertLessEqual(1, loss(y, p).numpy())  # Dice should be at 1 and cross-entropy can go to infinity
            self.assertEqual(0, round(loss(y, y).numpy(), 5))
        
        def test_ClDice(self):
            img = np.stack([np.zeros((5,5))]*2, axis=-1)
            skel = img.copy()
            skel[:, 2, :] = [0,1]
            y = tf.expand_dims(tf.convert_to_tensor(skel, dtype=tf.float32), axis=0)
            p = tf.expand_dims(tf.convert_to_tensor(img,  dtype=tf.float32), axis=0)
            
            for m in range(3):
                cldice = ClDice(iters=5, ndim=2, mode=m, smooth=1e-5)
                coef = cldice.coefficient
                loss = cldice.loss

                # coef
                self.assertEqual(0, round(coef(y, p).numpy(), 5))
                self.assertEqual(1, coef(y, y).numpy())

                # loss
                self.assertEqual(1, round(loss(y, p).numpy(), 5))
                self.assertEqual(0, loss(y, y).numpy())
        
        def test_DiceClDice(self):
            img = np.stack([np.zeros((5,5)), np.zeros((5,5))], axis=-1)
            skel = img.copy()
            skel[:, 2, :] = [0,1]
            y =  tf.expand_dims(tf.convert_to_tensor(skel,   dtype=tf.float32), axis=0)
            p1 = tf.expand_dims(tf.convert_to_tensor(1-img,  dtype=tf.float32), axis=0)
            p2 = tf.expand_dims(tf.convert_to_tensor(1-skel, dtype=tf.float32), axis=0)
            
            for m in range(3):
                dicecldice = DiceClDice(alpha=0.5, iters=5, ndim=2, mode=m, smooth=1e-5)
                coef = dicecldice.coefficient
                loss = dicecldice.loss

                # coef
                self.assertTrue(0 < coef(y, p1).numpy() < 1)
                self.assertEqual(0, round(coef(y, p2).numpy(), 5))
                self.assertEqual(1, coef(y, y).numpy())

                # loss
                self.assertTrue(0 < loss(y, p1).numpy() < 1)
                self.assertEqual(1, round(loss(y, p2).numpy(), 5))
                self.assertEqual(0, loss(y, y).numpy())
        
        def test_AdaWing(self):
            y = tf.convert_to_tensor([[0,1],[1,0]], dtype=tf.float32)
            p = tf.convert_to_tensor([[1,0],[0,1]], dtype=tf.float32)
            adawing = AdaptiveWingLoss()
            loss = adawing.loss

            # loss
            self.assertLessEqual(1, loss(y, p).numpy())
            self.assertEqual(0, loss(y, y).numpy())
        
        def test_ClDiceAdaWing(self):
            img = np.stack([np.zeros((5,5)), np.zeros((5,5))], axis=-1)
            skel = img.copy()
            skel[:, 2, :] = [0,1]
            y =  tf.expand_dims(tf.convert_to_tensor(skel,   dtype=tf.float32), axis=0)
            p1 = tf.expand_dims(tf.convert_to_tensor(1-img,  dtype=tf.float32), axis=0)
            p2 = tf.expand_dims(tf.convert_to_tensor(1-skel, dtype=tf.float32), axis=0)
            adawing = ClDiceAdaptiveWingLoss(ndim=2)
            loss = adawing.loss

            # loss
            self.assertGreaterEqual(round(loss(y, p2).numpy(), 5), 1)
            self.assertEqual(0, loss(y, y).numpy())

    unittest.main()