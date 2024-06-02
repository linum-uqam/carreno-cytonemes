# -*- coding: utf-8 -*-
import os
import training_unet
import numpy as np

# adjustable settings
training_unet.out_model_name = "unet2d_dec_unlabeled.h5"
training_unet.wdb_project    = 'unet2d_dec_unlabeled'

training_unet.params = {
    'ndim'     : 2,                  # 2 or 3
    'shape'    : [1, 192, 192],      # data shape
    'depth'    : 4,                  # unet depth
    'nfeat'    : 64,                 # nb feature for first conv layer
    'lr'       : [1e-4, 5e-5, 0.01], # learning rate [init, min, max]
    'warmup'   : 15,                 # nb epoch for lr warmup
    'decay'    : 15,                 # nb epoch for lr decay
    'bsize'    : 16,                 # batch size
    'nepoch'   : 40,                 # number of epoch
    'scaler'   : 'stand',            # "norm" or "stand"
    'label'    : 'soft',             # hard or soft input
    'sample'   : False,              # use sample weights for unbalanced data
    'weight'   : [0.37,38.54,4],     # use class weights for unbalanced data
    'order'    : 'after',            # where to put batch norm
    'ncolor'   : 3,                  # color depth for input
    'act'      : 'relu',             # activation
    'loss'     : 'cldiceadawing',    # loss function
    'topact'   : 'relu',             # top activation
    'dropout'  : 0.3,                # dropout rate
    'backbone' : 'vgg16',            # "unet" or "vgg16"
    'pretrn'   : True,               # pretrained encoder on imagenet
    'dupe'     : 16                  # nb d'usage pour un volume dans une époque
}


def setup_files(verbose=0):
    hard_label = training_unet.params['label'] == 'hard'

    # get file paths
    volumes = list(os.listdir(training_unet.config['VOLUME']['unlabeled']))
    sep1 = int(len(volumes) * 0.8)
    vol_train = volumes[:int(sep1)]
    sep2 = int(sep1 + np.ceil((len(volumes) - sep1) / 2))
    vol_valid = volumes[sep1:sep2]
    vol_test  = volumes[sep2:]
    fullpath = lambda dir, files : [os.path.join(dir, name) for name in files] * max(1, training_unet.params['dupe'])
    x_train = fullpath(training_unet.config['VOLUME']['rl_unlabeled'], vol_train)
    y_train = fullpath(training_unet.config['VOLUME']['unlabeled_target' if hard_label else 'unlabeled_soft_target'], vol_train)
    x_valid = fullpath(training_unet.config['VOLUME']['rl_unlabeled'], vol_valid)
    y_valid = fullpath(training_unet.config['VOLUME']['unlabeled_target' if hard_label else 'unlabeled_soft_target'], vol_valid)
    x_test =  fullpath(training_unet.config['VOLUME']['rl_unlabeled'], vol_test)
    y_test =  fullpath(training_unet.config['VOLUME']['unlabeled_target' if hard_label else 'unlabeled_soft_target'], vol_test)
    
    # sample weights
    w_train = fullpath(training_unet.config['VOLUME']['unlabeled_weight' if hard_label else 'unlabeled_soft_weight'], vol_train) if training_unet.params['sample'] else None
    w_valid = fullpath(training_unet.config['VOLUME']['unlabeled_weight' if hard_label else 'unlabeled_soft_weight'], vol_valid) if training_unet.params['sample'] else None

    if verbose:
        print("Training unlabeled dataset")
        if training_unet.params['sample']:
            print("-nb of instances :", len(x_train), "/", len(y_train), "/", len(w_train))
            print("Validation dataset")
            print("-nb of instances :", len(x_valid), "/", len(y_valid), "/", len(w_valid))
        else:
            print("-nb of instances :", len(x_train), "/", len(y_train))
            print("Validation dataset")
            print("-nb of instances :", len(x_valid), "/", len(y_valid))
        print("Testing dataset")
        print("-nb of instances :", len(x_test), "/",  len(y_test))
    return x_train, y_train, w_train, x_valid, y_valid, w_valid, x_test, y_test

training_unet.setup_files = setup_files
training_unet.main(verbose=1)