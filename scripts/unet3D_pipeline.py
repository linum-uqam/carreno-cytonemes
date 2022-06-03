# -*- coding: utf-8 -*-
import tifffile as tif
from skimage.morphology import binary_opening
import numpy as np
from carreno.segment.threshold import primary_object
from carreno.cytoneme.path import skeletonized_cell_paths, clean_cyto_paths
from carreno.io.tifffile import metadata
from carreno.io.csv import cells_info_csv
from carreno.utils.morphology import separate_blob, create_sphere

##############
# Parameters
##############
filename = "data/dataset/input/0.tif"  # path to an imagej tif volume with cell(s)
model = ...  # load model

def main():
    try:
        # assuming unit is um
        distances = metadata(filename)["axe_dist"]
        # convert scale to picometer like in imagej since selem is scaled for it
        distances = np.array(distances) * 1e6
    except:
        # if we don't have axis info, use default distances
        distances = np.array([0.26, 0.1201058, 0.1201058])
    
    # seperate volume in patches
    patch_shape = list(model.get_config()["layers"][0]["config"]["batch_input_shape"][1:])
    patch_jump = patch_shape.copy()
    patch_jump[1] = patch_jump[1] // 2
    patch_jump[2] = patch_jump[2] // 2
    
    patch, order = utils.patchify(volume, patch_shape, 2, patch_jump)
    
    # predict patch segmentation
    pred_patch = model.predict(np.array(patch))
    
    # reassemble patches into a volume
    pred_volume = utils.unpatchify(pred_patch, order, jump=patch_jump)
    
    # choose category based on softmax prediction
    final_volume = np.zeros_like(pred_volume)
    maximums = pred_volume.argmax(axis=-1)
    for i in range(pred_volume.shape[0]):
        for j in range(pred_volume.shape[1]):
            for k in range(pred_volume.shape[2]):
                final_volume[i, j, k, maximums[i, j, k]] = 1
    
    cells_info = []
    
    # label cytoneme
    cells_cyto = volume[..., 1]  # green
    cyto_lb = nd.label(cells_cyto)[0]
    
    # label body
    cells_body = volume[..., 2]  # blue
    body_lb = body_label(cells_body, distances=distances)
    
    # associate cytonemes to body
    association = associate_cytoneme(body_lb, cyto_lb)
    
    # skeletonize cytonemes
    cyto_sk = morphology.skeletonize_3d(volume[..., 1])  # range is [0, 255]
    cyto_lb[cyto_sk == 0] = 0
    
    for i in range(body_lb.max()):
        b = body_lb == i + 1  # body for label i+1
        
        # body metrics
        coords = np.where(b)
        body_z_start = np.amin(coords[0])
        body_z_end = np.amax(coords[0])
        
        # cytonemes metrics
        path, prob = cyto_paths(b, cyto_lb, association[i])
        
        # filter cytonemes
        filtered_path, filtered_prob = clean_cyto_paths(path, prob)
                
        # give result in a csv file
        cells_info.append({
            'body_z': [body_z_start, body_z_end],
            'path': filtered_path,
            'odds': filtered_prob
        })
    
    filename_csv = filename.rsplit( ".", 1 )[ 0 ] + '.csv'
    cells_info_csv(filename_csv, cells_info, distances)
    

if __name__ == "__main__":
    main()