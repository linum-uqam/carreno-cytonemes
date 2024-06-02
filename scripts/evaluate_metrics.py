# -*- coding: utf-8 -*-
import os

import utils
from carreno.pipeline.pipeline import Pipeline
from carreno.processing.transforms import Read

config = utils.get_config()
ilastik_dir = config['VOLUME']['ilastik']
tresh_dir   = config['VOLUME']['threshold']
unet2d_dir  = config['VOLUME']['unet2d']
unet3d_dir  = config['VOLUME']['unet3d']
target_dir  = config['VOLUME']['target']
p1, p2      = config['TRAINING']['evaluation']

def read_volume(dir):
    path1 = os.path.join(dir, p1 + ".tif")
    path2 = os.path.join(dir, p2 + ".tif")
    v1, v2, _ = Read()(path1, path2)
    return v1, v2

def main():
    pipeline = Pipeline()
    for dir in [ilastik_dir, tresh_dir, unet2d_dir, unet3d_dir, target_dir]:
        v1, v2 = read_volume(dir)
        pipeline.analyse(v1, os.path.join(dir, p1 + ".csv"))
        pipeline.analyse(v2, os.path.join(dir, p2 + ".csv"))

if __name__ == "__main__":
    main()
