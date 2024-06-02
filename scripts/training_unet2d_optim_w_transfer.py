# -*- coding: utf-8 -*-
import os
import training_unet
import tensorflow as tf

# local imports
from carreno.nn.unet import encoder_trainable
from carreno.nn.layers import ReluNormalization

# adjustable settings
training_unet.out_model_name = "unet2d_dec_optim_w_transfer.h5"
training_unet.wdb_project    = "unet2d_dec_optim_w_transfer"

training_unet.params = {
    'ndim'     : 2,                  # 2 or 3
    'shape'    : [1, 192, 192],      # data shape
    'depth'    : 4,                  # unet depth
    'nfeat'    : 64,                 # nb feature for first conv layer
    'lr'       : [1e-4, 5e-5, 0.001],# learning rate [init, min, max]
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
    'pretrn'   : False,              # pretrained encoder on imagenet
    'dupe'     : 48                  # nb d'usage pour un volume dans une époque
}

def setup_model(verbose=0):
    model_path = os.path.join(training_unet.config['DIR']['model'], "unet2d_dec_unlabeled.h5")
    print(f"- load model \"{model_path}\" ...")
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={"ReluNormalization": ReluNormalization},
                                       compile=False)
    # unfreeze encoder
    training_unet.encoder_trainable(model, True)
    if verbose:
        model.summary()
    return model

training_unet.setup_model = setup_model
training_unet.main(verbose=1)