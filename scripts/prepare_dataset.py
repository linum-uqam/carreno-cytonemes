# -*- coding: utf-8 -*-
import numpy as np
from shutil import rmtree, copy2
import pathlib
import tifffile as tif
import os
from pyunpack import Archive
import scipy.special
import scipy.ndimage
import unittest

# local imports
import utils
from carreno.io.fetcher import fetch_folder, folders_id
from carreno.utils.array import normalize
from carreno.processing.weights import balanced_class_weights

download                 = 0  # False if the folders are already downloaded
uncompress_raw           = 0  # uncompress archives in raw data, error if uncompressed files are missing with options uncompress_raw and hand_drawn_cyto
create_labeled_dataset   = 0  # organise uncompressed labeled data
create_sample_weights    = 0  # make weight distributions for labeled data patches (during `create_labeled_dataset`)
create_unlabeled_dataset = 0  # organise uncompressed unlabeled data
hand_drawn_cyto_dataset  = 0  # save hand drawn cytonemes (2D) in data
cleanup_uncompressed     = 0  # cleanup extracted files in raw folder

config        = utils.get_config()
raw_dir       = config['DIR']['raw']
drawing_dir   = config['DIR']['drawing']
download_dir  = config['DIR']['raw'].rsplit("/", 1)  # folder where downloads and dataset will be put, must not exist for download
input_dir     = config['VOLUME']['input']
target_dir    = config['VOLUME']['target']
weight_dir    = config['VOLUME']['weight']
starget_dir   = config['VOLUME']['soft_target']
sweight_dir   = config['VOLUME']['soft_weight']
unlabeled_dir = config['VOLUME']['unlabeled']
blur          = config['PREPROCESS']['blur']

# list of labeled volumes
# volumes = [[path_to_volume, volume_name]]
volumes = [
    ['Nouvelle annotation cellules/GFP #01.tif', 'ctrl1'],
    ['Nouvelle annotation cellules/GFP #02.tif', 'ctrl2'],
    ['GFP/GFP3.tif', 'ctrl3'],
    ['GFP/GFP4.tif', 'ctrl4'],
    ['Nouvelle annotation cellules/Slik GFP#01.tif', 'slik1'],
    ['Nouvelle annotation cellules/Slik GFP #02.tif', 'slik2'],
    ['Envoi annotation/Slik 3.tif', 'slik3'],
    ['Envoi annotation/Slik 4.tif', 'slik4'],
    ['Envoi annotation/Slik 5.tif', 'slik5'],
    ['Slik 6.tif', 'slik6']
]
volumes = [[os.path.join(raw_dir, path), name] for path, name in volumes]

# list of labeled cytonemes
cytonemes = [
    'Nouvelle annotation cellules/Mask cytoneme Ctrl GFP#1.tif',
    'Nouvelle annotation cellules/Mask cytoneme Ctrl GFP #02.tif',
    'Re annotation/Mask cytoneme GFP 3.tif',
    'Re annotation/Mask cytoneme GFP 4.tif',
    'Nouvelle annotation cellules/Mask cytoneme Slik GFP#1.tif',
    'Nouvelle annotation cellules/Mask cytoneme Slik GFP #02.tif',
    'Envoi annotation/Mask Cytoneme Slik3.tif',
    'Envoi annotation/Mask Cytoneme Slik4.tif',
    'Envoi annotation/Mask Cytoneme Slik5.tif',
    'Re annotation/Slik-6 deconvoluted-annotation-cytonemes.tif'
]
cytonemes = [os.path.join(raw_dir, path) for path in cytonemes]

# list of labeled cell bodies
bodies = [
    'Nouvelle annotation cellules/Mask cell body Ctrl GFP#1.tif',
    'Nouvelle annotation cellules/Mask cell body Ctrl GFP #02.tif',
    'Re annotation/Mask cell body GFP 3.tif',
    'Re annotation/Mask cell body GFP 4.tif',
    'Nouvelle annotation cellules/Mask cell body Slik GFP#1.tif',
    'Nouvelle annotation cellules/Mask cell body Slik GFP #02.tif',
    'Envoi annotation/Mask cell body Slik 3.tif',
    'Envoi annotation/Mask cell body Slik 4.tif',
    'Envoi annotation/Mask cell body Slik 5.tif',
    'Slik-6 deconvoluted-annotation-cell_body.tif'
]
bodies = [os.path.join(raw_dir, path) for path in bodies]

# list of unlabeled volumes
unlabeled_volumes = []
for (dir, dirnames, filenames) in os.walk(raw_dir + '/Non annotated Data'):
    for f in filenames:
        unlabeled_volumes.append(os.path.join(dir, f))

# list of hand drawn cytonemes in 2D for pipeline validation
# Expected format :
# data = [[cyto_path, volume_name, surname]]
data = [
    ["Ctrl GFP _Sample_1",  "Ctrl GFP _Sample_1.tif",  "gfp1-1"],
    ["Ctrl GFP 1_Sample_2", "Ctrl GFP 1_Sample_2.tif", "gfp1-2"],
    ["GFP 2_Sample_1",      "GFP 2_Sample_1.tif",      "gfp2-1"],
    ["GFP 2_Sample_2",      "GFP 2_Sample_2.tif",      "gfp2-2"],
    ["GFP 2_Sample_3",      "GFP 2_Sample_3.tif",      "gfp2-3"],
    ["GFP 3_Sample_1",      "GFP 3_Sample_1.tif",      "gfp3"],
    ["Slik GFP 1_Sample_1", "Slik GFP 1_Sample_1.tif", "slik1-1"],
    ["Slik GFP 1_Sample_2", "Slik GFP 1_Sample_2.tif", "slik1-2"],
    ["Slik GFP 2_Sample_1", "Slik GFP 2_Sample_1.tif", "slik2"],
    ["Slik GFP 3_Sample_1", "Slik GFP 3_Sample_1.tif", "slik3"],
    ["Slik GFP 4_Sample_1", "Slik GFP 4_Sample_1.tif", "slik4"],
]
for i in range(len(data)):
    data[i][0] = os.path.join(raw_dir, "Annotation Bon sens", data[i][0])
    data[i][1] = os.path.join(raw_dir, "Annotation Bon sens", data[i][1])


def download_raw_data(path):
    """Download carreno files from Google Drive
    Parameters
    ----------
    path : str, Path
        Path to save files into
    Returns
    -------
    None
    """
    if os.path.exists(path):
        print("Error : output folder already exists. Change the output folder or delete it if you do want to download.")
        exit()
    
    folders = folders_id()
    for name, id in folders.items():
        fetch_folder(id=id, output=path + "/" + name)
        
    return


def uncompress_files_in_folder(path):
    """Uncompressed zip and lar files in folder
    Parameters
    ----------
    path : str, Path
        Path to folder with files to uncompress within
    Returns
    -------
    None
    """
    for f in os.listdir(path):
        filename, extension = os.path.splitext(f)
    
        # god knows what a `.lar` file is, but that what I got so...
        if extension == ".zip" or extension == ".lar":
            # using pyunpack seems weird since this can be done with native
            # libs, but it seems my zip files aren't zip (Windows winrar)
            # and it's the only thing that works without changing files
            # extension.
            Archive(path + "/" + f).extractall(path)
    
    return


def create_dataset_input_folder(folder, volumes):
    """
    Create X folder using raw files from Basile
    Parameters
    ----------
    folder : str, Path
        Path where to create/override the input folder
    volumes : [Path, str]
        Path to tiff volumes with associated name
    Returns
    -------
    None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(len(volumes)):
        v = tif.imread(volumes[i][0])
        """ Changed input from int to float for image reconstruction
        # Basile said X inputs are meant to be 8 bits integer grayscale volumes
        x = normalize(v, 0, 255).astype(np.uint8)
        """
        x = normalize(v, 0, 1).astype(np.float32)
        tif.imwrite(folder + '/' + volumes[i][1] + '.tif', x, photometric='minisblack')
        

def create_dataset_target_folder(folder, volumes, cytonemes, bodies, blur=None):
    """
    Create Y folder using raw files from Basile
    Parameters
    ----------
    folder : str, Path
        Path where to create/override the target folder
    volumes : [Path]
        Path to tiff volumes
    cytonemes : [Path]
        Path to tiff cytonemes annotations
    bodies : [Path]
        Path to tiff bodies annotations
    blur : float, None
        sigma for gaussian filter (see SoftSeg article)
    Returns
    -------
    None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(len(volumes)):
        c = normalize(tif.imread(cytonemes[i]), 0, 1)
        b = normalize(tif.imread(bodies[i]), 0, 1)

        # Y targets are binary categorical volumes
        # saves memory compared to sparse categorical even though we need more than 1 channel since values are binary
        """
        For TiffFile, putting rgb photometric with bool dtype creates really weird artifacts.
        While it could have been nice to have `y` dtype as bool, that's just not an option for Tiff format.
        Plus bool dtype wouldn't work with SoftSeg (Soft ground-truth)
        """
        y = np.zeros([*(b.shape), 3], dtype=np.float32)
        y[..., 0] = np.logical_not(np.logical_or(c, b))
        y[..., 1] = c
        y[..., 2] = b

        # Soft ground-truth https://arxiv.org/ftp/arxiv/papers/2011/2011.09041.pdf
        if not blur is None:
            # blur axis less to respect anisotropic distances
            dist_ratio = 0.26 / 0.1201058
            y = scipy.ndimage.gaussian_filter(y, sigma=(blur/(dist_ratio), blur, blur, 0))
            
            # make sure everything is well distributed
            scipy.special.softmax(y, axis=-1)

        tif.imwrite(folder + '/' + volumes[i][1] + '.tif', y, photometric='rgb')


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
    
    instances = [0] * 3
    for f in filenames:
        # find classes instances
        y = tif.imread(os.path.join(target_folder, f))
        for i in range(len(instances)):
            instances[i] += y[..., i].sum()
        
    # get classes weights
    weights = balanced_class_weights(instances)

    for f in filenames:
        # save weighted volume
        y = tif.imread(os.path.join(target_folder, f))
        w_vol = np.sum(y * weights, axis=-1)
        tif.imwrite(os.path.join(folder, f), w_vol, photometric="minisblack")


def hand_drawn_cyto(drawing_path):
    """
    Delete all files in a folder which aren't zip or lar
    Parameters
    ----------
    drawing_path : str, Path
        Path to folder where to put hand drawn cytonemes
    Returns
    -------
    None
    """
    global data

    # remove patches folders if they already exist
    if os.path.isdir(drawing_path):
        rmtree(drawing_path)
    
    pathlib.Path(drawing_path).mkdir(parents=True, exist_ok=True)  # create folder

    for txt, img, association in data:
        path = drawing_path + "/" + association
        copy2(txt, path + ".txt")
        copy2(img, path + ".tif")

    return


def delete_uncompressed_files(path):
    """Delete all files in a folder which aren't zip or lar
    Parameters
    ----------
    path : str, Path
        Path to folder with non-compressed files to delete
    Returns
    -------
    None
    """
    for f in os.listdir(path):
        filename, extension = os.path.splitext(f)
        
        if extension != ".zip" and extension != ".lar" and f != "README.md":
            full_path = path + '/' + f
        
            # kill it in cold blood (hope it wasn't important)
            if os.path.isfile(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                rmtree(full_path)
    
    return


def main():
    # Get data
    if download:
        print("Downloading Google Drive files ...", end=" ")
        download_raw_data(download_dir)
        print("done")

    # Uncompress files
    if uncompress_raw:
        print("Uncompressing archives ...", end=" ")
        uncompress_files_in_folder(raw_dir)
        print("done")

    # Prepare dataset for supervised training
    if create_labeled_dataset:
        # this will override input and target folders so be careful
        print("Creating labeled dataset from raw data ...", end=" ")
        create_dataset_input_folder(input_dir, volumes)
        # hard labels
        create_dataset_target_folder(target_dir, volumes, cytonemes, bodies)
        # soft labels
        if blur:
            create_dataset_target_folder(starget_dir, volumes, cytonemes, bodies, blur=blur)
        print("done")
    
    if create_sample_weights:
        print("Creating weights from labeled dataset ...", end=" ")
        # weights for hard labels
        create_sample_weight_folder(weight_dir, target_dir)
        # weights for soft labels
        if blur:
            create_sample_weight_folder(sweight_dir, starget_dir)
        print("done")

    # Prepare dataset for self supervised training
    if create_unlabeled_dataset:
        # this will override input and target folders so be careful
        print("Creating unlabeled dataset from raw data ...", end=" ")
        #filenames = [str(i) for i in range(len(unlabeled_volumes))]
        filenames = [os.path.splitext(os.path.basename(fn))[0] for fn in unlabeled_volumes]  # get filenames w/ extensions (very long)
        create_dataset_input_folder(unlabeled_dir, list(zip(unlabeled_volumes, filenames)))
        print("done")

    # Copy hand drawn annotations
    if hand_drawn_cyto_dataset:
        print("Copying hand drawn annotations ...", end=" ")
        hand_drawn_cyto(drawing_dir)
        print("done")

    # Cleanup uncompressed files
    if cleanup_uncompressed:
        print("Cleaning uncompressed files ...", end=" ")
        delete_uncompressed_files(raw_dir)

        # custom delete
        others = ["GFP 1_Sample_1.zip",
                  "GFP 1_Sample_2.zip",
                  "GFP 5_Sample_5.zip",
                  "Slik_Sample_3.zip",
                  "Slik_Sample_4-2.zip",
                  "Slik-4-1.zip"]
    
        for f in others:
            full_path = raw_dir + '/' + f
            
            # kill file in cold blood (hope it wasn't important)
            if os.path.isfile(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                rmtree(full_path)

        print("done")


if __name__ == "__main__":
    main()
    
    class TestDataset(unittest.TestCase):
        def test_shape(self):
            def listdir_w_path(path):
                return [os.path.join(path, f) for f in os.listdir(path)]

            # make sure volume shape is uniform throughout x, y and w
            volume_per_ctg = {}
    
            if os.path.exists(input_dir):
                volume_per_ctg['x'] = listdir_w_path(input_dir)
            
            if os.path.exists(target_dir):
                volume_per_ctg['y'] = listdir_w_path(target_dir)
            
            if os.path.exists(weight_dir):
                volume_per_ctg['w'] = listdir_w_path(weight_dir)
        
            if os.path.exists(starget_dir):
                volume_per_ctg['sy'] = listdir_w_path(target_dir)
            
            if os.path.exists(sweight_dir):
                volume_per_ctg['sw'] = listdir_w_path(sweight_dir)
            
            available = list(volume_per_ctg.keys())
            template = volume_per_ctg[available[0]]
            for i in range(1, len(available)):
                compared_volumes = volume_per_ctg[available[i]]
                for t, v in zip(template, compared_volumes):
                    x = tif.imread(t)
                    y = tif.imread(v)
                    valid = x.shape[:3] == y.shape[:3]
                    self.assertTrue(valid)
                    if not valid:
                        print("shape mismatch between template {} ({}) and {} ({})".format(t, x.shape, v, y.shape))
            
    unittest.main()
