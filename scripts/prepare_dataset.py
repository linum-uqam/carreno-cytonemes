# -*- coding: utf-8 -*-
import numpy as np
from shutil import rmtree, copy2
import pathlib
import tifffile as tif
import os
from pyunpack import Archive
from carreno.io.fetcher import fetch_folder, folders_id
from carreno.utils.util import normalize
from carreno.processing.patchify import patchify

download = 0  # False if the folders are already downloaded
uncompress_raw = 1  # uncompress archives in raw data, error if uncompressed files are missing with options uncompress_raw and hand_drawn_cyto
create_dataset = 0  # organise uncompressed data
create_patches = 0  # seperate volume into patches
hand_drawn_cyto = 1  # save hand drawn cytonemes (2D) in data
cleanup_uncompressed = 1  # cleanup extracted files in raw folder

output = "data"  # folder where downloads and dataset will be put
dataset_name = "dataset"
raw_path = output + '/raw'
input_folder   = output + "/" + dataset_name + "/input"
target_folder  = output + "/" + dataset_name + "/target"
drawing_folder = output + "/drawn_cyto"
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
    
    for i in range(len(volumes)):
        v = tif.imread(volumes[i][0])
        c = tif.imread(cytonemes[i])
        b = tif.imread(bodies[i])
        
        # Files validation
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
            tif.imwrite(input_folder + '/' + volumes[i][1] + '.tif', x)
            
            # Y targets are binary categorical volumes
            # saves memory compared to sparse categorical even though we need more than 1 channel since values are binary
            y = np.zeros([*(x.shape), 3], dtype=bool)
            y[..., 0] = np.logical_not(np.logical_or(c, b))
            y[..., 1] = c
            y[..., 2] = b
            tif.imwrite(target_folder + '/' + volumes[i][1] + '.tif', y)
    
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
    
    xp_path = []
    yp_path = []
    
    x_files = os.listdir(x_path)
    y_files = os.listdir(y_path)

    for i in range(len(x_files)):
        inc = 0
        x = tif.imread(x_path + "/" + x_files[i])
        y = tif.imread(y_path + "/" + y_files[i])

        # TODO patch_shape is wrong for x, find fancier way to do this maybe?
        xp, __ = patchify(x, patch_shape[:-1], 2, stride)
        yp, __ = patchify(y, patch_shape, 1, stride)

        for j in range(len(yp)):
            """ I'll stop filtering since I didn't test unbalance impact (and subjective)
            # filter meaningless patches
            threshold = np.prod(patch_shape[:-1]) * 0.05  # 5% of patch is body or cytonemes
            if yp[j][:,:,:,1:3].sum() >= threshold:  # enough body and cyto
                ...
            """
            x_name, _ = os.path.splitext(x_files[i])
            y_name, _ = os.path.splitext(y_files[i])
            xp_p = xp_folder + "/" + x_name + "_" + str(inc) + '.tif'  # x patches save path
            yp_p = yp_folder + "/" + y_name + "_" + str(inc) + '.tif'  # y patches save path
                
            tif.imwrite(xp_p, xp[j])
            xp_path.append(xp_p)
            tif.imwrite(yp_p, yp[j])
            yp_path.append(yp_p)
            inc += 1
                
    return xp_path, yp_path


def hand_drawn_cyto(raw_path, drawing_path):
    """Delete all files in a folder which aren't zip or lar
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
    #f = "Annotation a la main Slik et GFP"  # old
    f = "Annotation Bon sens"
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

    # remove patches folders if they already exist
    if os.path.isdir(drawing_path):
        rmtree(drawing_path)
    
    pathlib.Path(drawing_path).mkdir(parents=True, exist_ok=True)  # create folder

    for txt, img, association in data:
        path = drawing_path + "/" + association
        copy2(raw_path + "/" + f + "/" + txt, path + ".txt")
        copy2(raw_path + "/" + f + "/" + img, path + ".tif")

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
    if create_dataset:
        # this will override input and target folders so be careful
        print("Creating dataset from raw data ...", end=" ")
        create_dataset_folder(raw_path, input_folder, target_folder)
        print("done")

    # Create patches
    if create_patches:
        print("Volume division ...", end=" ")
        prepare_patches(input_folder, target_folder, input_patch_folder, target_patch_folder, patch_shape, stride=stride)
        print("done")
    
    # Copy hand drawn annotations
    if hand_drawn_cyto:
        print("Copying hand drawn annotations ...", end=" ")
        hand_drawn_cyto(raw_path, drawing_folder)
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