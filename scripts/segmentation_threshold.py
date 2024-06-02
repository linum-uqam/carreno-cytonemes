# -*- coding: utf-8 -*-
import os
import tifffile as tif
import utils
from carreno.pipeline.pipeline import Threshold

config        = utils.get_config()
input_dir     = config['VOLUME']['input']
output_dir    = config['VOLUME']['threshold']
psf = tif.imread(os.path.join(config['DIR']['psf'], "Averaged PSF.tif"))
inputs = list(os.listdir(input_dir))
fullpath = lambda dir, files : [os.path.join(dir, name) for name in files]
trn, vld, tst = utils.split_dataset(input_dir)
fullpath = lambda dir, files : [os.path.join(dir, name) for name in files]
x_test   = fullpath(input_dir, tst)
p_test   = fullpath(output_dir, tst)

pipeline = Threshold()

for i in range(len(x_test)):
  print("Predicting", i, x_test[i], "...")
  x = tif.imread(x_test[i])
  d = pipeline.restore(x,
                       psf=psf,
                       iteration=25,
                       nlm_size=8,
                       nlm_dist=1,
                       r=50,
                       md_size=[1, 3, 3],
                       butterworth=[0.5, 0.95, 0.09],
                       amount=8)
  p = pipeline.segmentation(d,
                            ro=1.5,
                            rc=2)
  tif.imwrite(p_test[i], p)
  print("- Result written at", p_test[i])