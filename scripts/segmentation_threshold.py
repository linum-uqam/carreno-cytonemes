# -*- coding: utf-8 -*-
import os
import tifffile as tif

import utils
from carreno.pipeline.pipeline import Threshold

config        = utils.get_config()
input_dir     = config['VOLUME']['input']
output_dir    = config['VOLUME']['threshold']

inputs = list(os.listdir(input_dir))
fullpath = lambda dir, files : [os.path.join(dir, name) for name in files]
xs = fullpath(input_dir, inputs)
ys = fullpath(output_dir, inputs)
psf = tif.imread(os.path.join(config['DIR']['psf'], "Averaged PSF.tif"))
pipeline = Threshold()

for i in range(len(ys)):
  
  print("Processing", i, xs[i], "...")
  
  x = tif.imread(xs[i])
  
  d = pipeline.restore(x,
    psf=psf,
    iteration=50,
    nlm_size=8,
    nlm_dist=1,
    r=50,
    md_size=[3, 5, 5],
    butterworth=[0.5, 0.95, 0.09],
    amount=8)
    
  p = pipeline.segmentation(d,
    ro=0.26,
    rc=1)
    
  tif.imwrite(ys[i], p)

  print("- Result written at", ys[i])