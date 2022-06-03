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
denoise = None  # denoise volume ("bm", "nlm", None)
psf = "data/psf/Averaged PSF.tif"  # another denoising option for applying richardson lucy filter
sharpen = True  # sharpens volume before segmentation with maximum gaussian 2e derivative

def main():
    try:
        # assuming unit is um
        distances = metadata(filename)["axe_dist"]
        # convert scale to picometer like in imagej since selem is scaled for it
        distances = np.array(distances) * 1e6
    except:
        # if we don't have axis info, use default distances
        distances = np.array([0.26, 0.1201058, 0.1201058])
        
    cells_info = []
    
    # get cell(s) mask
    cell_m = primary_object(tif.imread(filename),
                            denoise=denoise,
                            psf=tif.imread(psf),
                            sharpen=sharpen)
    
    # get cell(s) body mask
    sphere = create_sphere(2, distances)
    body_m = binary_opening(cell_m, selem=sphere)
    
    # separate cells and get cells bodies
    bodies, cells = separate_blob(body_m,
                                  cell_m,
                                  distances=distances)
    
    for i in range(len(cells)):
        # body metrics
        coords = np.where(bodies[i] > 0)
        body_z_start = np.amin(coords[0])
        body_z_end = np.amax(coords[0])
        
        # cytonemes metrics
        path, prob = skeletonized_cell_paths(bodies[i], cells[i])
        
        # filter cytonemes
        filtered_path, filtered_prob = clean_cyto_paths(path, prob)
                
        # give result in a csv file
        cells_info.append({'body_z': [body_z_start, body_z_end],
                           'path': filtered_path,
                           'odds': filtered_prob})
    
    filename_csv = filename.rsplit(".", 1)[0] + '.csv'
    cells_info_csv(filename_csv, cells_info, distances)


if __name__ == "__main__":
    main()