# -*- coding: utf-8 -*-
import os
import training_unet
import numpy as np
from carreno.nn.unet import UNet, encoder_trainable

# adjustable settings
training_unet.out_model_name = "unet2d_dec_optim.h5"
training_unet.wdb_project    = 'unet2d_dec_optim'

params = {
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
training_unet.params = params

def setup_model(verbose=0):
    model = UNet(shape=params['shape'][1:] + [params['ncolor']],
                 n_class=training_unet.config['PREPROCESS']['n_cls'],
                 depth=params['depth'],
                 n_feat=params['nfeat'],
                 dropout=params['dropout'],
                 norm_order=params['order'],
                 activation=params['act'],
                 top_activation=params['topact'],
                 backbone=None if params['backbone'] == "unet" else params['backbone'],
                 pretrained=params['pretrn'])

    # freeze encoder
    if params['pretrn']:
        encoder_trainable(model, True)

    if verbose:
        model.summary()
        
    return model

training_unet.setup_model = setup_model
training_unet.main(verbose=1)