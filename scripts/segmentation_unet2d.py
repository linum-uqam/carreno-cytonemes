# -*- coding: utf-8 -*-
import os
import tifffile as tif
import numpy as np

import utils
from carreno.pipeline.pipeline import UNet2D

config        = utils.get_config()
input_dir     = config['VOLUME']['input']
output_dir    = config['VOLUME']['unet2d']
model_path    = "TODO"
patch_size    = [192, 192]
stride        = [96] * 2
sigma         = 10

inputs = list(os.listdir(input_dir))
fullpath = lambda dir, files : [os.path.join(dir, name) for name in files]
xs = fullpath(input_dir, inputs)
ys = fullpath(output_dir, inputs)
pipeline = UNet2D(model_path)

for i in range(len(ys)):
  
  print("Processing", i, xs[i], "...")
  
  x = tif.imread(xs[i])
  sigma = 10.0     # width of kernel

  xax = np.arange(-patch_size[0], patch_size[0], 1)
  zax = np.arange(-patch_size[1], patch_size[1], 1)
  xx, zz = np.meshgrid(xax, zax)
  w = np.exp(-(xx**2 + zz**2) / (2*sigma**2))
  
  p = pipeline.segmentation(x, stride, w)
    
  tif.imwrite(ys[i], p)

  print("- Result written at", ys[i])