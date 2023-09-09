# -*- coding: utf-8 -*-
import os
import training_unet2d_base

# adjustable settings
training_unet2d_base.out_model_name = "unet2d_optim.h5"
training_unet2d_base.wdb_project    = 'unet2d_optim'

training_unet2d_base.params = {
    'ndim'     : 2,              # 2 or 3
    'shape'    : [1, 192, 192],  # data shape
    'depth'    : 4,              # unet depth
    'nfeat'    : 64,             # nb feature for first conv layer
    'lr'       : 0.01,           # learning rate
    'bsize'    : 4,              # batch size
    'nepoch'   : 100,            # number of epoch
    'scaler'   : 'stand',        # "norm" or "stand"
    'label'    : 'soft',         # hard or soft input
    'weight'   : True,           # use class weights for unbalanced data
    'order'    : 'after',        # where to put batch norm
    'ncolor'   : 3,              # color depth for input
    'act'      : 'relu',         # activation
    'loss'     : 'cldiceadawing',# loss function
    'topact'   : 'relu',         # top activation
    'dropout'  : 0.4,            # dropout rate
    'backbone' : 'vgg16',        # "unet" or "vgg16"
    'pretrn'   : True,           # pretrained encoder on imagenet
    'slftrn'   : False,          # pretrained encoder on unlabeled
    'dupe'     : 16              # nb d'usage pour un volume dans une époque
}

training_unet2d_base.main(verbose=1)