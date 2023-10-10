# -*- coding: utf-8 -*-
import tensorflow as tf
import os

# local imports
import utils
from carreno.nn.unet import UNet

# adjustable settings
input_ndim = 4  # 5 for 3D
depth      = 4
n_features = 64
dropout    = 0.3
backbone_i = 1
activation = 'relu'


def main(verbose=0):
    config = utils.get_config()
    backbone = config['TRAINING']['backbone'][backbone_i]

    # must add color channel to grayscale
    input_shape = config['PREPROCESS']['patch'][1:] + [3] if input_ndim == 4 else config['PREPROCESS']['patch'] + [1]
    model_name  = "untrn_unet{}D-{}-{}-{}-{}-{}.h5".format(input_ndim-2, depth, n_features, dropout, activation, backbone)
    model_path  = os.path.join(config['TRAINING']['output'], model_name)
    
    unet = UNet(shape=input_shape,
                n_class=config['PREPROCESS']['n_cls'],
                depth=depth,
                n_feat=n_features,
                dropout=dropout,
                activation=activation,
                backbone=backbone,
                pretrained=True)
    
    if verbose:
        unet.summary()

    print("Saving model to {} ... ".format(model_path), end="")
    unet.save(model_path)
    print("done")


if __name__ == "__main__":
    main(verbose=1)