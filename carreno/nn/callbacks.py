# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from pathlib import Path

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

# TODO custom callback for confusion matrix on epoch end
# https://stackoverflow.com/questions/52285501/how-can-i-create-a-custom-callback-in-keras