# -*- coding: utf-8 -*-
import os
import tifffile as tif
import tensorflow as tf
import numpy as np

import utils
from carreno.pipeline.pipeline import UNet2D
from carreno.nn.metrics import Dice, ClDice
from carreno.nn.layers import ReluNormalization
from carreno.processing.transforms import Compose, Read, Standardize, Squeeze, Stack
from carreno.utils.array import gaussian_kernel

config = utils.get_config()

# model
#model_path = os.path.join(config['DIR']['model'], "unet2d_optim.h5")
model_path = "E:/tmp_linum/unet2d_dec_optim_w_transfer.h5"
model = tf.keras.models.load_model(model_path,
                                   custom_objects={"ReluNormalization": ReluNormalization},
                                   compile=False)
pipeline = UNet2D(model)

# test files
input_dir     = config['VOLUME']['rl_input']
target_dir    = config['VOLUME']['target']
output_dir    = config['VOLUME']['unet2d']
trn, vld, tst = utils.split_dataset(input_dir)
fullpath = lambda dir, files : [os.path.join(dir, name) for name in files]
x_test   = fullpath(input_dir, tst)
y_test   = fullpath(target_dir, tst)
p_test   = fullpath(output_dir, tst)

# patchify
patch_size = [192, 192, 3]
sigma = 10
w = gaussian_kernel(patch_size[:2], sigma=10)
w = np.stack([w]*3, axis=-1)

# augmentation
test_aug = Compose(transforms=[
  Read(),
  Standardize(),
  Squeeze(axis=0, p=0),
  Stack(axis=-1, n=3)
])

# evaluation metrics
iters  = 10
ndim   = 3
pad    = 2
smooth = 1e-6
dice   = Dice(smooth=smooth).coefficient
cldice = ClDice(iters=iters, ndim=ndim, mode=pad, smooth=smooth).coefficient

for i in range(len(y_test)):
  print("Predicting", i, x_test[i], "...")
  x, y, _ = test_aug(x_test[i], y_test[i])
  stride = list((x.shape[1:-1]) - np.array(patch_size[:-1])) + [3]
  p = pipeline.segmentation(x, stride, w)

  ty = tf.expand_dims(tf.convert_to_tensor(y, dtype=tf.float32), 0)
  tp = tf.expand_dims(tf.convert_to_tensor(p, dtype=tf.float32), 0)
  print("- Dice score", dice(ty, tp))
  print("- clDice score", cldice(ty, tp))
  print("- Result written at", p_test[i])
  print()
  tif.imwrite(p_test[i], p)
