# -*- coding: utf-8 -*-
import numpy as np
from shutil import rmtree
import pathlib
import tifffile as tif
import os
from pyunpack import Archive
from carreno.io.fetcher import fetch_folder, folders_id
from carreno.utils.util import normalize
from carreno.utils.patchify import patchify

download = True  # False if the folders are already downloaded
create_dataset = True  # uncompress and organise downloads
create_patches = True  # seperate volume into patches
cleanup_uncompressed = True  # cleanup extracted files in raw folder

output = "data"  # folder where downloads and dataset will be put
dataset_name = "dataset"
raw_path = output + '/raw'
input_folder  = output + "/" + dataset_name + "/input"
target_folder = output + "/" + dataset_name + "/target"
patch_shape = (64, 64, 64, 3)
stride = None
input_patch_folder = output + "/" + dataset_name + "/input_p"
target_patch_folder = output + "/" + dataset_name + "/target_p"


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


def create_dataset_folder(raw_path, input_folder, target_folder):
    """Create x and y folders using raw files from Basile
    Parameters
    ----------
    raw_path : str, Path
        Path to folder with uncompressed raw files
    input_folder : str, Path
        Path where to create/override the input folder
    target_folder : str, Path
        Path where to create/override the target folder
    Returns
    -------
    None
    """
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    volumes = [
        raw_path + '/Nouvelle annotation cellules/GFP #01.tif',
        raw_path + '/Nouvelle annotation cellules/GFP #02.tif',
        raw_path + '/GFP/GFP3.tif',
        raw_path + '/GFP/GFP4.tif',
        raw_path + '/Nouvelle annotation cellules/Slik GFP#01.tif',
        raw_path + '/Nouvelle annotation cellules/Slik GFP #02.tif',
        raw_path + '/Envoi annotation/Slik 3.tif',
        raw_path + '/Envoi annotation/Slik 4.tif',
        raw_path + '/Envoi annotation/Slik 5.tif',
        raw_path + '/Slik 6.tif'
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
    
    # Files validation
    for i in range(len(volumes)):
        v = tif.imread(volumes[i])
        c = tif.imread(cytonemes[i])
        b = tif.imread(bodies[i])
        
        if v.shape[0:3] != c.shape[0:3]:
            print("Error : wrong cytonemes annotation shape for", volumes[i])
            print("- volume", volumes[i], v.shape)
            print("- cytonemes", cytonemes[i], c.shape)
        elif v.shape[0:3] != b.shape[0:3]:
            print("Error : wrong bodies annotation shape for", volumes[i])
            print("- volume", volumes[i], v.shape)
            print("- bodies", cytonemes[i], c.shape)
        else:
            # X inputs are 8 bits integer grayscale volumes
            # Basile told me volumes are meant to be 8 bits and other format could create artifacts
            x = normalize(v, 0, 255).astype(np.uint8)
            tif.imwrite(input_folder + '/' + str(i) + '.tif', x)
            
            # Y targets are binary categorical volumes
            # saves memory compared to sparse categorical even though we need more than 1 channel since values are binary
            y = np.zeros([*(x.shape), 3], dtype=bool)
            y[..., 0] = np.logical_not(np.logical_or(c, b))
            y[..., 1] = c
            y[..., 2] = b
            tif.imwrite(target_folder + '/' + str(i) + '.tif', y)
    
    return


def prepare_patches(x_path, y_path, xp_folder, yp_folder, patch_shape, stride=None):
    """Divide dataset in patches and save
    Parameters
    ----------
    x_path : list
        TIF volumes paths
    y_path : list
        TIF annotated volumes paths
    xp_folder : Path
        Folder path for saving X patches (overwrites!)
    yp_folder : Path
        Folder path for saving Y patches (overwrites!)
    patch_shape : list
        Patch desired shape
    stride : list
        Jump between patches, default to patch shape
    Returns
    -------
    xp_path : list
        TIF patches paths
    yp_path : list
        TIF patches paths
    """
    # remove patches folders if they already exist
    if os.path.isdir(xp_folder):
        rmtree(xp_folder)
    
    if os.path.isdir(yp_folder):
        rmtree(yp_folder)
    
    pathlib.Path(xp_folder).mkdir(parents=True, exist_ok=True)  # create folder
    pathlib.Path(yp_folder).mkdir(parents=True, exist_ok=True)
    
    inc = 0
    xp_path = []
    yp_path = []
    
    x_files = os.listdir(x_path)
    y_files = os.listdir(y_path)

    for i in range(len(x_files)):
        x = tif.imread(x_path + "/" + x_files[i])
        y = tif.imread(y_path + "/" + y_files[i])

        # TODO patch_shape is wrong for x, find fancier way to do this
        xp, __ = patchify(x, patch_shape[:-1], 2, stride)
        yp, __ = patchify(y, patch_shape, 1, stride)

        for j in range(len(yp)):
            """ I'll stop filtering since I didn't test unbalance impact (and subjective)
            # filter meaningless patches
            threshold = np.prod(patch_shape[:-1]) * 0.05  # 5% of patch is body or cytonemes
            if yp[j][:,:,:,1:3].sum() >= threshold:  # enough body and cyto
                ...
            """
            xp_p = xp_folder + "/" + str(inc) + '.tif'  # x patches save path
            yp_p = yp_folder + "/" + str(inc) + '.tif'  # y patches save path
                
            tif.imwrite(xp_p, xp[j])
            xp_path.append(xp_p)
            tif.imwrite(yp_p, yp[j])
            yp_path.append(yp_p)
            inc += 1
                
    return xp_path, yp_path


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

    # Prepare dataset for supervised training
    if create_dataset:
        print("Uncompressing archives ...", end=" ")
        uncompress_files_in_folder(raw_path)
        print("done")

        # this will override input and target folders so be careful
        print("Creating dataset from raw data ...", end=" ")
        create_dataset_folder(raw_path, input_folder, target_folder)
        print("done")

    # Create patches
    if create_patches:
        print("Volume division ...", end=" ")
        prepare_patches(input_folder, target_folder, input_patch_folder, target_patch_folder, patch_shape, stride=stride)
        print("done")

    # Cleanup uncompressed files
    if cleanup_uncompressed:
        print("Cleaning uncompressed files ...", end=" ")
        delete_uncompressed_files(raw_path)
        print("done")


if __name__ == "__main__":
    main()