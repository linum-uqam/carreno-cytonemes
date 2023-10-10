# -*- coding: utf-8 -*-
import os
import training_unet3d_base

# adjustable settings
training_unet3d_base.out_model_name = "unet3d_optim.h5"
training_unet3d_base.wdb_project    = 'unet3d_optim'

training_unet3d_base.params = {
    'ndim'     : 3,              # 2 or 3
    'shape'    : [16, 128, 128], # data shape
    'depth'    : 4,              # unet depth
    'nfeat'    : 64,             # nb feature for first conv layer
    'lr'       : 0.001,          # learning rate
    'bsize'    : 2,              # batch size
    'nepoch'   : 100,            # number of epoch
    'scaler'   : 'stand',        # "norm" or "stand"
    'label'    : 'soft',         # hard or soft input
    'weight'   : True,           # use class weights for unbalanced data
    'order'    : 'after',        # where to put batch norm
    'ncolor'   : 3,              # color depth for input
    'act'      : 'relu',         # activation
    'loss'     : 'cldiceadawing',# loss function
    'topact'   : 'relu',         # top activation
    'dropout'  : 0.3,            # dropout rate
    'backbone' : 'vgg16',        # "unet" or "vgg16"
    'pretrn'   : True,           # pretrained encoder on imagenet
    'slftrn'   : False,          # pretrained encoder on unlabeled
    'dupe'     : 16              # nb d'usage pour un volume dans une époque
}

training_unet3d_base.main(verbose=1)