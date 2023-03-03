# -*- coding: utf-8 -*-
import numpy as np
from shutil import rmtree, copy2
import pathlib
import tifffile as tif
import os
from pyunpack import Archive
import scipy

from carreno.io.fetcher import fetch_folder, folders_id
from carreno.utils.util import normalize
from carreno.processing.patches import patchify, reshape_patchify
from carreno.processing.weights import balanced_class_weights

download =                  0  # False if the folders are already downloaded
uncompress_raw =            0  # uncompress archives in raw data, error if uncompressed files are missing with options uncompress_raw and hand_drawn_cyto
create_labelled_dataset =   0  # organise uncompressed labelled data
create_unlabelled_dataset = 0  # organise uncompressed unlabelled data
create_patches =            0  # seperate volume into patches
create_sample_weights =     1  # make weight distributions for labelled data patches
hand_drawn_cyto_dataset =   0  # save hand drawn cytonemes (2D) in data
cleanup_uncompressed =      0  # cleanup extracted files in raw folder

output =                   "data"  # folder where downloads and dataset will be put, must not exist for download
dataset_name =             "dataset"
raw_path =                 output + '/raw'
input_folder   =           output + "/" + dataset_name + "/input"
target_folder  =           output + "/" + dataset_name + "/target"
soft_target_folder =       output + "/" + dataset_name + "/soft_target"
unlabelled_folder =        output + "/" + dataset_name + "/unlabelled"
drawing_folder =           output + "/drawn_cyto"
input_patch_folder =       output + "/" + dataset_name + "/input_p"
target_patch_folder =      output + "/" + dataset_name + "/target_p"
weight_patch_folder =      output + "/" + dataset_name + "/weight_p"
soft_target_patch_folder = output + "/" + dataset_name + "/soft_target_p"
unlabelled_patch_folder =  output + "/" + dataset_name + "/unlabelled_p"
patch_shape = [48, 96, 96]
stride = None
blur = 1.5

# volumes = [[path_to_volume, volume_name]]
volumes = [
    [raw_path + '/Nouvelle annotation cellules/GFP #01.tif', 'ctrl1'],
    [raw_path + '/Nouvelle annotation cellules/GFP #02.tif', 'ctrl2'],
    [raw_path + '/GFP/GFP3.tif', 'ctrl3'],
    [raw_path + '/GFP/GFP4.tif', 'ctrl4'],
    [raw_path + '/Nouvelle annotation cellules/Slik GFP#01.tif', 'slik1'],
    [raw_path + '/Nouvelle annotation cellules/Slik GFP #02.tif', 'slik2'],
    [raw_path + '/Envoi annotation/Slik 3.tif', 'slik3'],
    [raw_path + '/Envoi annotation/Slik 4.tif', 'slik4'],
    [raw_path + '/Envoi annotation/Slik 5.tif', 'slik5'],
    [raw_path + '/Slik 6.tif', 'slik6']
]

cytonemes = [
    raw_path + '/Nouvelle annotation cellules/Mask cytoneme Ctrl GFP#1.tif',
    raw_path + '/Nouvelle annotation cellules/Mask cytoneme Ctrl GFP #02.tif',
    raw_path + '/Re annotation/Mask cytoneme GFP 3.tif',
    raw_path + '/Re annotation/Mask cytoneme GFP 4.tif',
    raw_path + '/Nouvelle annotation cellules/Mask cytoneme Slik GFP#1.tif',
    raw_path + '/Nouvelle annotation cellules/Mask cytoneme Slik GFP #02.tif',
    raw_path + '/Envoi annotation/Mask Cytoneme Slik3.tif',
    raw_path + '/Envoi annotation/Mask Cytoneme Slik4.tif',
    raw_path + '/Envoi annotation/Mask Cytoneme Slik5.tif',
    raw_path + '/Re annotation/Slik-6 deconvoluted-annotation-cytonemes.tif'
]

bodies = [
    raw_path + '/Nouvelle annotation cellules/Mask cell body Ctrl GFP#1.tif',
    raw_path + '/Nouvelle annotation cellules/Mask cell body Ctrl GFP #02.tif',
    raw_path + '/Re annotation/Mask cell body GFP 3.tif',
    raw_path + '/Re annotation/Mask cell body GFP 4.tif',
    raw_path + '/Nouvelle annotation cellules/Mask cell body Slik GFP#1.tif',
    raw_path + '/Nouvelle annotation cellules/Mask cell body Slik GFP #02.tif',
    raw_path + '/Envoi annotation/Mask cell body Slik 3.tif',
    raw_path + '/Envoi annotation/Mask cell body Slik 4.tif',
    raw_path + '/Envoi annotation/Mask cell body Slik 5.tif',
    raw_path + '/Slik-6 deconvoluted-annotation-cell_body.tif'
]

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
    data[i][0] = raw_path + "/Annotation Bon sens/" + data[i][0]
    data[i][1] = raw_path + "/Annotation Bon sens/" + data[i][1]


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
    volumes : [Path]
        Path to tiff volumes
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
            for axis in range(y.shape[-1]):
                y[..., axis] = scipy.ndimage.gaussian_filter(y[..., axis], sigma=blur)
        
            # make sure everything is well distributed
            scipy.special.softmax(y, axis=-1)

        tif.imwrite(folder + '/' + volumes[i][1] + '.tif', y, photometric='rgb')


def prepare_patches(volume_path, patch_folder, patch_shape, stride=None, mode=1):
    """
    Divide dataset in patches and save
    Parameters
    ----------
    volume_path : list
        TIF volumes paths
    patch_folder : Path
        Folder path for saving patches (overwrites!)
    patch_shape : list
        Patch desired shape
    stride : list
        Jump between patches, default to patch shape
    mode : int
        Interpolation mode (1:nearest-neighbor, 2:bilinear, 3:bicubic)
    Returns
    -------
    p_path : list
        TIF patches paths
    """
    # remove patches folders if they already exist
    if os.path.isdir(patch_folder):
        rmtree(patch_folder)
    
    # create folder
    pathlib.Path(patch_folder).mkdir(parents=True, exist_ok=True)
    
    p_path = []
    files = os.listdir(volume_path)

    for i in range(len(files)):
        inc = 0
        v = tif.imread(volume_path + "/" + files[i])
        p = patchify(v, patch_shape=patch_shape, stride=stride, resize_mode=mode)
        p, __ = reshape_patchify(p, len(patch_shape))

        for j in range(len(p)):
            """ I'll stop filtering since I didn't test unbalance impact (and subjective)
            # filter meaningless patches
            threshold = np.prod(patch_shape[:-1]) * 0.05  # 5% of patch is body or cytonemes
            if yp[j][:,:,:,1:3].sum() >= threshold:  # enough body and cyto
                ...
            """
            name, _ = os.path.splitext(files[i])
            p_path.append(patch_folder + "/" + name + "_" + str(inc) + '.tif')  # patches save path
            tif.imwrite(p_path[-1], p[j])
            inc += 1
            
    return p_path


def create_sample_weight_folder(folder, target_folder):
    """
    Create W folder using Y folder
    Parameters
    ----------
    folder : str, Path
        Path where to create/override the weighted patches folder
    target_folder : str, Path
        Path where annotation patches are
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
    
    for f in filenames:
        # find classes instances
        instances = [0] * 3
        y = tif.imread(os.path.join(target_folder, f))
        for i in range(len(instances)):
            instances[i] += y[..., i].sum()
        
        # get classes weights
        weights = balanced_class_weights(instances)

        # save weighted volume
        w_vol = np.sum(y * weights, axis=-1)
        tif.imwrite(os.path.join(folder, f), w_vol, photometric="minisblack")


def hand_drawn_cyto(drawing_path):
    """
    Delete all files in a folder which aren't zip or lar
    Parameters
    ----------
    raw_path : str, Path
        Path to folder with raw data
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
        download_raw_data(output)
        print("done")

    # Uncompress files
    if uncompress_raw:
        print("Uncompressing archives ...", end=" ")
        uncompress_files_in_folder(raw_path)
        print("done")

    # Prepare dataset for supervised training
    if create_labelled_dataset:
        # this will override input and target folders so be careful
        print("Creating labelled dataset from raw data ...", end=" ")
        create_dataset_input_folder(input_folder, volumes)
        create_dataset_target_folder(target_folder, volumes, cytonemes, bodies)
        if blur:
            create_dataset_target_folder(soft_target_folder, volumes, cytonemes, bodies, blur=blur)
        print("done")
    

    if create_unlabelled_dataset:
        # this will override input and target folders so be careful
        print("Creating unlabelled dataset from raw data ...", end=" ")
        create_dataset_input_folder(unlabelled_folder, [])
        print("done")

    # Create patches
    if create_patches:
        print("Volume division ...", end=" ")
        if os.path.exists(input_folder):
            prepare_patches(input_folder, input_patch_folder, patch_shape, stride=stride, mode=2)
        if os.path.exists(target_folder):
            prepare_patches(target_folder, target_patch_folder, patch_shape+[3], stride=stride, mode=1)
        if os.path.exists(soft_target_folder):
            prepare_patches(soft_target_folder, soft_target_patch_folder, patch_shape+[3], stride=stride, mode=1)
        if os.path.exists(unlabelled_folder):
            prepare_patches(unlabelled_folder, unlabelled_folder, patch_shape, stride=stride, mode=2)
        print("done")
    
    if create_sample_weights and os.path.exists(target_patch_folder):
        # this will override weight folder so be careful
        print("Creating sample weights for patches ...", end=" ")
        create_sample_weight_folder(weight_patch_folder, target_patch_folder)
        print("done")

    # Copy hand drawn annotations
    if hand_drawn_cyto_dataset:
        print("Copying hand drawn annotations ...", end=" ")
        hand_drawn_cyto(drawing_folder)
        print("done")

    # Cleanup uncompressed files
    if cleanup_uncompressed:
        print("Cleaning uncompressed files ...", end=" ")
        delete_uncompressed_files(raw_path)

        # custom delete
        others = ["GFP 1_Sample_1.zip",
                  "GFP 1_Sample_2.zip",
                  "GFP 5_Sample_5.zip",
                  "Slik_Sample_3.zip",
                  "Slik_Sample_4-2.zip",
                  "Slik-4-1.zip"]
    
        for f in others:
            full_path = raw_path + '/' + f
            
            # kill it in cold blood (hope it wasn't important)
            if os.path.isfile(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                rmtree(full_path)

        print("done")


if __name__ == "__main__":
    main()