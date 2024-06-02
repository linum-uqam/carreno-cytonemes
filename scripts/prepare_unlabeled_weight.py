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
import scipy.special
import scipy.ndimage
from shutil import rmtree, copy2
import pathlib
from carreno.utils.array import normalize
from carreno.processing.weights import balanced_class_weights
import matplotlib.pyplot as plt

infos = {}
with open("config.yml", 'r') as file:
  infos = yaml.safe_load(file)

inputs = list(os.listdir(infos['VOLUME']['unlabeled']))
fullpath = lambda dir, files : [os.path.join(dir, name) for name in files]
xs = fullpath("data/dataset/unlabeled_target", inputs)
ys = fullpath("data/dataset/unlabeled_soft_target", inputs)
ws = fullpath("data/dataset/unlabeled_soft_weight", inputs)

def create_sample_weight_folder(folder, target_folder):
    """
    Create W folder using Y folder
    Parameters
    ----------
    folder : str, Path
        Path where to create/override the weighted volumes folder
    target_folder : str, Path
        Path where annotation volumes are
    Returns
    -------
    None
    """
    # remove patches folders if they already exist
    if os.path.isdir(folder):
        rmtree(folder)
    
    # create folder
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    filenames = []
    for f in os.listdir(target_folder):
        filenames.append(f)
        
    # get classes weights
    weights = np.array([0.37, 38.54, 4])
    print(weights)

    test = True

    for f in filenames:
        # save weighted volume
        y = tif.imread(os.path.join(target_folder, f))
        w_vol = np.sum(y * weights, axis=-1)
        
        if test:
            test=False
            plt.imshow(w_vol[10]); plt.show()
        
        tif.imwrite(os.path.join(folder, f), w_vol, photometric="minisblack")


print("Starting")
print("--------")
blur = 1.5

for i in range(len(ys)):
  
  print("blur", i, xs[i], " ... ")
  
  x = tif.imread(xs[i])
  
  dist_ratio = 0.26 / 0.1201058
  y = scipy.ndimage.gaussian_filter(x, sigma=(blur/(dist_ratio), blur, blur, 0))
            
  # make sure everything is well distributed
  y = (1 / y.sum(axis=-1))[..., np.newaxis] * y
    
  tif.imwrite(ys[i], y)

print("Starting weight")
create_sample_weight_folder("data/dataset/unlabeled_soft_weight", "data/dataset/unlabeled_soft_target")