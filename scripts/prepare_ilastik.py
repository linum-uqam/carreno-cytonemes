# -*- coding: utf-8 -*-
import os
import tifffile as tif
import numpy as np
from pathlib import Path

import utils
from carreno.processing.categorical import categorical_to_sparse

config = utils.get_config()
target_dir  = config['VOLUME']['target']
ilastik_dir = config['VOLUME']['ilastik_target']

def main():
    Path(ilastik_dir).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(target_dir):
        if f[-4:] != ".tif":
            # skip non-tiff files
            continue
        in_path = os.path.join(target_dir, f)
        print('Converting "{}" to ilastik format ... '.format(in_path), end="")
        soft_pred = tif.imread(in_path)
        hard_pred = categorical_to_sparse(soft_pred).astype(np.uint8)
        # tif format only imports 2 classes, no issues with ndarray
        out_path = os.path.join(ilastik_dir, f)[:-4] + ".npy"
        np.save(out_path, hard_pred)
        print('done at', out_path)
    
if __name__ == "__main__":
    main()
