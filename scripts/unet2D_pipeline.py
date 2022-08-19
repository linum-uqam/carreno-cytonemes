# -*- coding: utf-8 -*-
import tensorflow as tf
import tifffile as tif
import numpy as np
import scipy.ndimage as nd
import os
from skimage.morphology import skeletonize_3d
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from pathlib import Path

from carreno.cytoneme.path import skeletonized_cyto_paths, clean_cyto_paths
from carreno.io.tifffile import metadata
from carreno.io.csv import cells_info_csv
from carreno.processing.patchify import volume_pred_from_img

data_folder = "data"  # folder where downloads and dataset will be put
dataset_folder = data_folder + "/dataset"
output_folder  = data_folder + "/output"
model_path = data_folder + "/model/test.h5"
filename  = dataset_folder + "/input/0.tif"  # path to an imagej tif volume with cell(s)
csv_output = output_folder + "/" + filename.rsplit(".", 1)[0] + '.csv'

# Replace or merge with seperate_blob implementations in carreno.utils.morphology
def seperate_blobs(x, min_dist=10, distances=[1, 1, 1]):
    """Separate blobs using watershed. Seeds are found using the foreground pixels distance from background pixels.
    Parameters
    ----------
    x : list, ndarray
        binary mask of blobs
    min_dist : float
        minimum distance between seeds for watershed
    distances : list, ndarray
        axis distances in order
    Returns
    -------
    label : ndarray
        labelled blob
    """
    # find 1 local max per blob
    distance = nd.distance_transform_edt(x)
    coords = peak_local_max(distance,
                            min_distance=min_dist,
                            labels=x > 0)
    
    # seperate the cells
    local_max = np.zeros(distance.shape, dtype=bool)
    local_max[tuple(coords.T)] = True
    markers = nd.label(local_max)[0]
    label = watershed(-distance, markers, mask=x)

    return label


# TODO similar idea to skeletonized_cell_paths in threshold pipeline, but with distance
def associate_cytoneme(body_label, cyto_label):
    """Matches cytonemes with the nearest body.
    TODO express ambiguity
    Parameters
    ----------
    body_label : ndarray
        labelled bodies
    cyto_label : ndarray
        labelled cytonemes
    Returns
    -------
    association : list
        list for each body containing a list of associated cytonemes
    """
    hv = 10e3  # high value
    body_mask = body_label > 0
    body_dist = nd.distance_transform_edt(body_mask)
    body_dist[body_dist == 0] = hv
    
    # 1 list of associated cytonemes per body
    association = []
    for i in range(body_label.max()):
        association.append([])
    
    # fill association
    for lb in range(1, cyto_label.max() + 1):
        cyto = cyto_label == lb
        cyto_dist = nd.distance_transform_edt(np.logical_not(cyto))
        cyto_dist[cyto_dist == 0] = hv
        
        # closest point on a body TODO consider all min for handling cyto intersections
        depth, row, line = np.unravel_index(np.argmin(body_dist * cyto_dist), body_mask.shape)
        
        association[body_label[depth, row, line] - 1].append(lb)
    
    return association


def main():
    try:
        # assuming unit is um
        distances = metadata(filename)["axe_dist"]
        # convert scale to picometer like in imagej since selem is scaled for it
        distances = np.array(distances) * 1e6
    except:
        # if we don't have axis info, use default distances
        distances = np.array([0.26, 0.1201058, 0.1201058])
    
    model = tf.keras.models.load_model(model_path, compile=False)
    volume = tif.imread(filename)
    
    # seperate volume in patches
    patch_shape = list(model.get_config()["layers"][0]["config"]["batch_input_shape"][1:])
    stride = [1] + [i // 2 for i in patch_shape[:-1]]
    
    # predict volume
    pred = volume_pred_from_img(model, volume, stride=stride)
    
    # choose category based on softmax prediction
    final_volume = np.zeros_like(pred)
    maximums = pred.argmax(axis=-1)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            for k in range(pred.shape[2]):
                final_volume[i, j, k, maximums[i, j, k]] = 1
    
    cells_info = []
    
    # label cytoneme
    cells_cyto = final_volume[..., 1]  # green
    cyto_lb = nd.label(cells_cyto)[0]
    
    # label body
    cells_body = final_volume[..., 2]  # blue
    body_lb = seperate_blobs(cells_body, distances=distances)

    # associate cytonemes to body
    association = associate_cytoneme(body_lb, cyto_lb)
    
    # skeletonize cytonemes
    cyto_sk = skeletonize_3d(final_volume[..., 1])  # range is [0, 255]
    cyto_lb[cyto_sk == 0] = 0
    
    for i in range(body_lb.max()):
        b = body_lb == i + 1  # body for label i+1
        
        # body metrics
        coords = np.where(b)
        body_z_start = np.amin(coords[0])
        body_z_end = np.amax(coords[0])
        
        # cytonemes metrics
        path, prob = skeletonized_cyto_paths(b, cyto_lb, association[i])
        
        # filter cytonemes
        filtered_path, filtered_prob = clean_cyto_paths(path, prob)
                
        cells_info.append({'body_z': [body_z_start, body_z_end],
                           'path': filtered_path,
                           'odds': filtered_prob})
    
    # save results in a csv file
    # save test prediction if we want to check it out more
    folder = os.path.dirname(csv_output)
    Path(folder).mkdir(parents=True, exist_ok=True)
    cells_info_csv(csv_output, cells_info, distances)
    

if __name__ == "__main__":
    main()