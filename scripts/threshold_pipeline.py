import tifffile as tif
from skimage.morphology import binary_opening
import csv
import numpy as np
from carreno.segment.threshold import primary_object
from carreno.cytoneme.path import skeletonized_cell_paths, clean_cyto_paths
from carreno.utils.morphology import separate_blob, create_sphere
from carreno.utils.util import euclidean_dist

filename = "data/dataset/input/0.tif"  # path to a tif volume with cell(s)
distances = [1, 1, 1]  # axis distances in order
denoise = None  # denoise volume ("bm", "nlm", None)
psf = "data/psf/Averaged PSF.tif"  # another denoising option for applying richardson lucy filter
sharpen = True  # sharpens volume before segmentation with maximum gaussian 2e derivative


def cells_info_csv(filename, cells_info, distances):
    """Write paths info for one cell into a csv file
    Parameters
    ----------
    filename : path, str
        location to save csv file
    cells_info : dict
        contains all the metric we want to save
    distances : list, ndarray
        axis distances in order for calculating length between coordinates
    """
    with open(filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        
        # add headers
        header = ['cell_id',
                  'body_z_start',
                  'body_z_end',
                  'path_id',
                  'x',
                  'y',
                  'z',
                  'dot_product',
                  'length']
        writer.writerow(header)
        
        for cell_id in range(len(cells_info)):
            # add data
            for i in range(len(cells_info[cell_id]['path'])):
                path_id = i + 1  # start id from 1
                length = 0
                
                n = len(cells_info[cell_id]['path'][i])
                for j in range(1, n):
                    data = [cell_id,
                            cells_info[cell_id]['body_z'][0],
                            cells_info[cell_id]['body_z'][1],
                            path_id,
                            cells_info[cell_id]['path'][i][0][2],
                            cells_info[cell_id]['path'][i][0][1],
                            cells_info[cell_id]['path'][i][0][0],
                            round(cells_info[cell_id]['odds'][i][0], 4),
                            round(length, 4)]
                    writer.writerow(data)
                    
                    # update path length
                    if j + 1 < n:
                        length += euclidean_dist(cells_info[cell_id]['path'][i][j],
                                                       cells_info[cell_id]['path'][i][j+1],
                                                       distances)
    
    print('Results written in', filename)
    
    return


def main():
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
        cells_info.append({
            'body_z': [body_z_start, body_z_end],
            'path': filtered_path,
            'odds': filtered_prob
        })
    
    filename_csv = filename.rsplit( ".", 1 )[0] + '.csv'
    cells_info_csv(filename_csv, cells_info, distances)


if __name__ == "__main__":
    main()