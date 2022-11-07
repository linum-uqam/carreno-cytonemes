# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from carreno.processing.classify import categorical_multiclass, categorical_to_sparse

def early_stop(metric, mode='min', patience=5, verbose=1):
    callback = tf.keras.callbacks.EarlyStopping(monitor=metric,
                                                mode=mode,
                                                patience=patience,
                                                verbose=verbose)

    return callback


def model_checkpoint(path, metric='loss', mode='min', verbose=1):
    folder = os.path.dirname(path)
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                  monitor=metric,
                                                  mode=mode,
                                                  save_best_only=True,
                                                  verbose=verbose)

    return callback

# https://stackabuse.com/custom-keras-callback-for-saving-prediction-on-each-epoch-with-visualizations/
class confusion_matrix_history(tf.keras.callbacks.Callback):
    def __init__(self, train_generator, validation_generator, n_class=2):
        self.tcm = []
        self.vcm = []
        self.tgen = train_generator
        self.vgen = validation_generator
        self.n_class = n_class
        self._tcm = np.zeros([n_class]*2, dtype=int)
        self._vcm = np.zeros([n_class]*2, dtype=int)
    

    def on_train_batch_end(self, batch, logs=None):
        # sum the confusion matrix of each training batch
        b = self.tgen.batch  # [[xs], [ys], [ws]]
        xs, ys = b[:2]
        
        preds = self.model(xs)
        classified_preds = categorical_multiclass(preds)
        
        # sklearn.confusion_matrix doesn't take kindly to categorical data
        sparse_y = categorical_to_sparse(ys)
        sparse_p = categorical_to_sparse(classified_preds)
        
        # sklearn.confusion_matrix doesn't take kindly more than 3 axis
        flat_y = sparse_y.flatten()
        flat_p = sparse_p.flatten()
        
        cm = confusion_matrix(y_true=flat_y, y_pred=flat_p, labels=np.arange(1, self.n_class+1))
        self._tcm + cm
        
        return


    def on_test_batch_end(self, batch, logs=None):
        # sum the confusion matrix of each validation batch
        b = self.vgen.batch  # [[xs], [ys], [ws]]
        xs, ys = b[:2]
        
        preds = self.model(xs)
        classified_preds = categorical_multiclass(preds)
        
        # sklearn.confusion_matrix doesn't take kindly to categorical data
        sparse_y = categorical_to_sparse(ys)
        sparse_p = categorical_to_sparse(classified_preds)
        
        # sklearn.confusion_matrix doesn't take kindly more than 3 axis
        flat_y = sparse_y.flatten()
        flat_p = sparse_p.flatten()
        
        cm = confusion_matrix(y_true=flat_y, y_pred=flat_p, labels=np.arange(1, self.n_class+1))
        self._vcm += cm
        
        return
    
    def on_epoch_end(self, epoch, logs=None):
        # add confusion matrix for the epoch
        self.tcm.append(self._tcm)
        self.vcm.append(self._vcm)
        
        # reset sums of confusion matrix for next epoch
        self._tcm = 0
        self._vcm = 0