# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import albumentations as A
import volumentations as V
import matplotlib.pyplot as plt

# local imports
import utils
from carreno.nn.layers import model2D_to_3D

# adjustable settings
inp_model_name = "slfspv_1e-05-untrn_unet2D-4-64-0.3-relu-VGG16.h5"
out_model_name = "tfr3D_{}.h5".format(inp_model_name.rsplit('.', 1)[0])


def main(verbose=0):
    config = utils.get_config()

    # load model
    path_to_unet = os.path.join(config['TRAINING']['output'], inp_model_name)
    model2D = tf.keras.models.load_model(path_to_unet, compile=False)

    model3D = model2D_to_3D(model2D, 48)

    if verbose:
        model3D.summary()

    out_model_path = os.path.join(config['TRAINING']['output'], out_model_name)
    print("Saving model to {} ... ".format(out_model_path), end="")
    model3D.save(out_model_path)
    print("done")


if __name__ == "__main__":
    main(verbose=1)