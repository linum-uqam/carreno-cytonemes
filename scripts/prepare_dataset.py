import numpy as np
import imageio as io
import os
from pyunpack import Archive
from carreno.io.fetcher import fetch_folder, folders_id
from carreno.utils import normalize

folders = folders_id()
download = False  # False if the folders are already downloaded
output = "data"
dataset_name = "dataset"
input_folder  = output + "/" + dataset_name + "/input"
target_folder = output + "/" + dataset_name + "/target"

anno_bodies = [
    output + '/raw/GFP1-2&SLIK1-2/Mask cell body Slik1.tif',
    output + '/raw/GFP1-2&SLIK1-2/Mask cell body Slik2.tif',
    output + '/raw/GFP1-2&SLIK1-2/Mask cell body Ctrl1.tif',
    output + '/raw/GFP1-2&SLIK1-2/Mask cell body Ctrl2.tif',
    output + '/raw/GFP/Mask cell body Ctrl3.tif',
    output + '/raw/GFP/Mask cell body Ctrl4.tif',
    output + '/raw/SLIK/Mask cell body Slik3.tif',
    output + '/raw/SLIK/Mask cell body Slik4.tif',
    output + '/raw/SLIK/Mask cell body Slik5.tif',
    output + '/raw/Slik6/Slik-6 deconvoluted-annotation-cell_body.tif'
]

anno_cytos = [
    output + '/raw/GFP1-2&SLIK1-2/Mask cytoneme Slik1.tif',
    output + '/raw/GFP1-2&SLIK1-2/Mask cytoneme Slik2.tif',
    output + '/raw/GFP1-2&SLIK1-2/Mask cytoneme Ctrl1.tif',
    output + '/raw/GFP1-2&SLIK1-2/Mask cytoneme Ctrl2.tif',
    output + '/raw/GFP/Mask cytoneme Ctrl3.tif',
    output + '/raw/GFP/Mask cytoneme Ctrl4.tif',
    output + '/raw/SLIK/Mask Cytoneme Slik3.tif',
    output + '/raw/SLIK/Mask Cytoneme Slik4.tif',
    output + '/raw/SLIK/Mask Cytoneme Slik5.tif',
    output + '/raw/Slik6/Slik-6 deconvoluted-annotation-cytonemes.tif'
]

# Get data
if download:
    if os.path.exists(output):
        print("Error : output folder already exists. Change the output folder or delete it if you do want to download.")
        exit()
    
    for name, id in folders.items():
        print("Downloading `", name, "` folder ...", sep="")
        fetch_folder(id=id, output=output + "/" + name)

# Prepare dataset for supervised training
if not os.path.exists(input_folder):
    os.makedirs(input_folder)
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

raw_path = output + '/raw'
"""
## cheating since one of the files as a bullshit extension and I believe it's just a zip file
bullshit = raw_output + "/Slik 6 annotation.lar"
if os.path.exists(bullshit):
    os.rename(bullshit,
              raw_output + "/Slik 6 annotation.zip")
"""

not_uncompressed_files = []
for f in os.listdir(raw_path):
    filename, extension = os.path.splitext(f)
    # god knows what a `.lar` file is, but that what I got so...
    if extension == ".zip" or extension == ".lar":
        # using pyunpack seems weird since this can be done with native
        # libs, but it seems my zip files aren't zip and it's the only
        # that works without changing files extension.
        Archive(raw_path + "/" + f).extractall(raw_path)
    not_uncompressed_files.append(f)

volumes = [
    raw_path + '/Nouvelle annotation cellules/GFP #01.tif',
    raw_path + '/Nouvelle annotation cellules/GFP #02.tif',
    raw_path + '/GFP/GFP3.tif',
    raw_path + '/GFP/GFP4.tif',
    raw_path + '/Nouvelle annotation cellules/Slik GFP#01.tif',
    raw_path + '/Nouvelle annotation cellules/Slik GFP #02.tif',
    raw_path + '/Envoi annotation/Slik3.tif',
    raw_path + '/Envoi annotation/Slik4.tif',
    raw_path + '/Envoi annotation/Slik5.tif',
    raw_path + '/Slik6.tif'
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
    v = io.imread(volumes[i])
    c = io.imread(cytonemes[i])
    b = io.imread(bodies[i])
    
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
        x = normalize(v, 0, 255).astype(np.uint8)
        io.imread(input_folder + '/' + str(i) + '.tif', x)
        
        # Y targets are binary categorical volumes
        # saves memory compared to sparse categorical even though we need more than 1 channel since values are binary
        y = np.zeros([*(x.shape), 3], dtype=bool)
        y[..., 0] = np.logical_not(np.logical_or(c, b))
        y[..., 1] = c
        y[..., 2] = b
        io.imread(target_folder + '/' + str(i) + 'tif', y)
        
# Cleanup uncompressed files
for f in os.listdir(raw_path):
    if not f in not_uncompressed_files:
        # remove file
        ...
        