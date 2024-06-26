# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from pathlib import Path
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback

def early_stop(metric, mode='min', patience=5, verbose=1):
    callback = tf.keras.callbacks.EarlyStopping(monitor=metric,
                                                mode=mode,
                                                patience=patience,
                                                restore_best_weights=True,
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