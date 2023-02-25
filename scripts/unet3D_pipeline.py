# -*- coding: utf-8 -*-
import tensorflow as tf
import tifffile as tif
import numpy as np
from skimage.transform import resize

from carreno.cell.path import extract_metric
from carreno.io.tifffile import metadata
from carreno.processing.patches import volume_pred_from_vol

data_folder = "data"  # folder where downloads and dataset will be put
dataset_folder = data_folder + "/dataset"
output_folder  = data_folder + "/output"
filename       = dataset_folder + "/input/slik3.tif"  # path to an imagej tif volume with cell(s)
model_path     = output_folder + "/model/unet3D.h5"

fname = lambda path : path.split("/")[-1].split(".", 1)[0]  # get filename without extension
csv_output     = output_folder + "/" + fname(filename) + '_' + fname(model_path) + ".csv"


def main():
    try:
        # assuming unit is um
        distances = metadata(filename)["axe_dist"]
        # convert scale to picometer like in imagej since selem is scaled for it
        distances = np.array(distances) * 1e6
    except:
        # if we don't have axis info, use default distances
        distances = np.array([0.26, 0.1201058, 0.1201058])
    
    model = tf.keras.models.load_model(model_path, compile=False)
    volume = tif.imread(filename)
    
    # seperate volume in patches
    volume_patch_shape = list(model.get_config()["layers"][0]["config"]["batch_input_shape"][1:-1])
    stride = [i // 2 for i in volume_patch_shape]

    # predict volume
    pred = volume_pred_from_vol(model, volume, stride=stride)
    
    # keep volume shape which might have changed with patchify
    pred = resize(pred,
                  output_shape=volume.shape,
                  order=1,
                  preserve_range=True,
                  anti_aliasing=False)
    
    # pass prediction trough pipeline
    extract_metric(pred=pred,
                   csv_output=csv_output,
                   distances=distances)
    

if __name__ == "__main__":
    main()