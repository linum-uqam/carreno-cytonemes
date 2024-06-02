import yaml
import os
import tifffile as tif
import numpy as np
import random
from skimage.restoration import estimate_sigma
from sklearn.model_selection import train_test_split
from pathlib import Path
import carreno.processing.transforms as tfs
from carreno.pipeline.pipeline import Threshold
import carreno.processing.categorical

print("current folder", os.getcwd())

infos = {}
with open("config.yml", 'r') as file:
  infos = yaml.safe_load(file)

inputs = list(os.listdir(infos['VOLUME']['unlabeled']))
fullpath = lambda dir, files : [os.path.join(dir, name) for name in files]
xs = fullpath(infos['VOLUME']['unlabeled'], inputs)
ys = fullpath("data/dataset/unlabeled_target", inputs)
psf = tif.imread(os.path.join(infos['DIR']['psf'], "Averaged PSF.tif"))
pipeline = Threshold()

print("Starting")
print("--------")

for i in range(len(ys)):
  
  print("processing", i, ys[i], " ... ")
  
  x = tif.imread(xs[i])
  
  d = pipeline.restore(x,
    psf=psf,
    iteration=25,
    nlm_size=1,
    nlm_dist=8,
    r=50,
    amount=8,
    md_size=[1, 3, 3],
    butterworth=[0.5, 0.95, 0.09],)
  
  p = pipeline.segmentation(d,
    ro=1.5,
    rc=0.5)
    
  tif.imwrite(ys[i], p)