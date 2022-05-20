import numpy as np
from shutil import rmtree
import tifffile as tif
import os
from pyunpack import Archive
from carreno.io.fetcher import fetch_folder, folders_id
from carreno.utils.util import normalize

download = False  # False if the folders are already downloaded
create_dataset = True  # use downloaded folders
cleanup_uncompressed = True  # cleanup extracted files in raw folder

output = "data"  # folder where downloads and dataset will be put
dataset_name = "dataset"
input_folder  = output + "/" + dataset_name + "/input"
target_folder = output + "/" + dataset_name + "/target"

# Get data
if download:
    print("Downloading Google Drive files ...", end=" ")
    if os.path.exists(output):
        print("Error : output folder already exists. Change the output folder or delete it if you do want to download.")
        exit()
    
    folders = folders_id()
    for name, id in folders.items():
        print("-Downloading `", name, "` folder ...", sep="")
        fetch_folder(id=id, output=output + "/" + name)
    print("done")

# Prepare dataset for supervised training
if create_dataset:
    raw_path = output + '/raw'
    
    print("Uncompressing archives ...", end=" ")
    for f in os.listdir(raw_path):
        filename, extension = os.path.splitext(f)
        # god knows what a `.lar` file is, but that what I got so...
        if extension == ".zip" or extension == ".lar":
            # using pyunpack seems weird since this can be done with native
            # libs, but it seems my zip files aren't zip and it's the only
            # that works without changing files extension.
            Archive(raw_path + "/" + f).extractall(raw_path)
    print("done")
    
    print("Creating dataset from raw data ...", end=" ")
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
    print("done")

# Cleanup uncompressed files
if cleanup_uncompressed:
    print("Cleaning uncompressed files ...", end=" ")
    for f in os.listdir(raw_path):
        filename, extension = os.path.splitext(f)
        if extension != ".zip" and extension != ".lar" and f != "README.md":
            full_path = raw_path + '/' + f
            # kill it in cold blood
            if os.path.isfile(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                rmtree(full_path)
    print("done")