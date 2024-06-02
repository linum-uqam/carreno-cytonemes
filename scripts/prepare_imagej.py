# -*- coding: utf-8 -*-
"""
Purposes of this script :
- This script creates the volumes I want to open via ImageJ for visualization.
- ImageJ usually works better with intensity range between 0-255.
  I often have issues where intensity aren't normalized by default.
  In which case, we barely see anything except darkness.
"""
import os
import tifffile as tif
import imageio as io
import numpy as np
import pandas as pd
from skimage.segmentation import watershed
from skimage.morphology import skeletonize_3d
from scipy import ndimage as nd

import utils
from carreno.cell.body import associate_cytoneme
from carreno.utils.morphology import seperate_blobs
from carreno.processing.transforms import Read
from carreno.processing.categorical import categorical_multiclass


config = utils.get_config()
ilastik_dir = config['VOLUME']['ilastik']
tresh_dir   = config['VOLUME']['threshold']
unet2d_dir  = config['VOLUME']['unet2d']
unet3d_dir  = config['VOLUME']['unet3d']
target_dir  = config['VOLUME']['target']
predictions = config['TRAINING']['evaluation']
input_dir   = config['VOLUME']['input']
manual_dir  = config['DIR']['drawing']


def read_volume(dir):
    path1 = os.path.join(dir, predictions[0] + ".tif")
    path2 = os.path.join(dir, predictions[1] + ".tif")
    v1, v2, _ = Read()(path1, path2)
    return categorical_multiclass(v1), categorical_multiclass(v2)


def create_class_volume(path, x):
    # Creates a volume to easily see classes in 3d for imagej.
    y = np.zeros_like(x, dtype=np.uint8)
    y[x[..., 1]] = [0,255,0]
    y[x[..., 2]] = [0,0,255]
    tif.imwrite(path, y)


def create_false_neg_volume(path, x, y):
    # Creates a volume of the false positives for cyto and body.
    # Helps to see what went wrong with the segmentation.
    fn = np.clip(y.copy().astype(float) - x, 0, 1)
    fn[fn > 0] = 255
    fn[..., 0] = 0  # don't keep false positives for background
    tif.imwrite(path, fn.astype(np.uint8))


def create_cell_seperation(path, x, cytos, bodies, associations):
    # Creates a volume to easily see the seperation between 2
    # neighbouring cells. Helps to see how much we messed up the
    # seperation and what kind of scenario can happen.
    y = np.zeros_like(x, dtype=np.uint8)
    n = len(associations)
    lbs = np.linspace(50, 255, n, dtype=np.uint8)[::-1]
    for i in range(len(associations)):
        lb = lbs[i]
        y[bodies == i+1] = lb
        for j in associations[i]:
            y[cytos == j+1] = lb
    tif.imwrite(path, y)


def create_cyto_lost(path, skelly, csv):
    # Creates a volume of the cytoneme we kept post processing
    # vs the cytonemes that were segmented.
    # Helps to see how aggressively we filter out paths.
    selem = [
        [[0,0,0],
         [0,1,0],
         [0,0,0]],
        [[0,1,0],
         [1,1,1],
         [0,1,0]],
        [[0,0,0],
         [0,1,0],
         [0,0,0]],
    ]
    expected = skelly.copy()
    expected = nd.binary_dilation(expected, structure=selem, iterations=1)
    found = np.zeros_like(skelly, dtype=bool)
    for i in csv.index:
        found[csv['z'][i], csv['y'][i], csv['x'][i]] = True
    found = nd.binary_dilation(found, structure=selem, iterations=1)
    y = np.zeros(list(expected.shape) + [3], dtype=np.uint8)
    y[expected, 1] = 255
    y[found, 0] = 255
    tif.imwrite(path, y)


def create_cyto_manual(path, csv, lost, img):
    # add annoted start/end point of cytonemes
    # on a projection view of input volume
    y = img.copy()
    lost_cyto = np.max(lost[..., 0] > 0, axis=0)
    kept_cyto = np.max(lost[..., 1] > 0, axis=0)
    print("Number of cyto ratio lost", lost_cyto.sum() / kept_cyto.sum())
    y[lost_cyto, 0] = 200
    y[kept_cyto, 1] = 200
    selem = [[0,1,0],
             [1,1,1],
             [0,1,0]]
    id, xi, yi, xf, yf, length = csv.columns
    for i, d_row in enumerate(csv.iloc):
        manual_pt = np.zeros(img.shape[:-1], dtype=bool)
        xiv = int(d_row[xi])
        yiv = int(d_row[yi])
        xfv = int(d_row[xf])
        yfv = int(d_row[yf])
        manual_pt[yiv, xiv] = True
        manual_pt[yfv, xfv] = True
        manual_pt = nd.binary_dilation(manual_pt, structure=selem, iterations=2)
        #rnd_color = (np.random.rand(3,) * 255).astype(np.uint8)
        y[manual_pt] = [255, 0, 0]
    io.imwrite(path, np.flip(y, axis=0))


def main():
    for dir in [tresh_dir, unet2d_dir, unet3d_dir, target_dir]:
        print("Setting up", dir, "... ", end="")
        # classes
        volumes = read_volume(dir)
        for p, v in zip(predictions, volumes):
            create_class_volume(os.path.join(dir, p + "_class.tif"), v)
        # classes false negative
        targets = read_volume(target_dir)
        for p, v, t in zip(predictions, volumes, targets):
            ...#create_false_neg_volume(os.path.join(dir, p + "_class_fn.tif"), v, t)
        # cells seperation
        skellies = [skeletonize_3d(v[..., 1]) for v in volumes]  # range is [0, 255]
        cyto_lbs = [nd.label(cyto_sk, structure=np.ones([3,3,3]))[0] for cyto_sk in skellies]  # skeleton cyto label
        cyto_rgs = [watershed(cyto,
                              markers=cyto,
                              mask=v[..., 1]) for v, cyto in zip(volumes, cyto_lbs)]  # full cyto label
        body_lbs = [seperate_blobs(v[..., 2], distances=config["PREPROCESS"]["distance"]) for v in volumes]
        associations = [associate_cytoneme(body_lb, cyto_lb) for body_lb, cyto_lb in zip(body_lbs, cyto_lbs)]  # associate cytonemes to cell
        for i in range(len(volumes)):
            create_cell_seperation(os.path.join(dir, predictions[i] + "_asso.tif"), volumes[i], cyto_rgs[i], body_lbs[i], associations[i])
        # cyto we kept vs lost
        csvs = [pd.read_csv(os.path.join(dir, p + ".csv")) for p in predictions]
        for i in range(len(csvs)):
            create_cyto_lost(os.path.join(dir, predictions[i] + "_lost_cyto.tif"), skellies[i], csvs[i])
        print("done")


    print("Setting up projection view of manual cyto ... ", end="")
    csvs = [pd.read_csv(os.path.join(manual_dir, manual_path), sep="\t") for manual_path in ["gfp1-1.txt", "slik1-1.txt"]]
    csvs = [pd.concat([first_body, pd.read_csv(os.path.join(manual_dir, manual_path), sep="\t")]) \
                   for first_body, manual_path in zip(csvs, ["gfp1-2.txt", "slik1-2.txt"])]
    lost_cytos = [tif.imread(os.path.join(target_dir, pred + "_lost_cyto.tif")) for pred in predictions]
    imgs = [tif.imread(os.path.join(manual_dir, img)) for img in ["gfp1-1.tif", "slik1-1.tif"]]
    for p, csv, lost, img in zip(predictions, csvs, lost_cytos, imgs):
        create_cyto_manual(os.path.join(target_dir, p + "_manual_cyto.png"), csv, lost, img)
    print("done")
    
    
if __name__ == "__main__":
    main()
