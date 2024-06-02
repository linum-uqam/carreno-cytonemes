# -*- coding: utf-8 -*-
import csv
from carreno.utils.array import euclidean_dist


def cells_info_csv(filename, cells_info, distances, ndigits=4):
    """Write paths info for one cell into a csv file
    Parameters
    ----------
    filename : path, str
        Location to save csv file
    cells_info : [map]
        Contains all the metric we want to save
        Expected map keys and values:
        - 'body_z' : [start_idx, end_idx]
        - 'cyto_to_cell' : float
        - 'path' : [cytoneme_path]
        - 'odds': [path_probability]
    distances : list, ndarray
        Axis distances in order for calculating length between coordinates
    ndigits : int
        Number of digits to show for floats in csv
    Returns
    -------
    None
    """
    with open(filename, 'w', encoding='UTF8') as f:
        display_float = lambda x: float(round(x, ndigits))
        writer = csv.writer(f)
        # add headers
        header = ['cell_id',
                  'body_z_start',
                  'body_z_end',
                  'cyto_to_cell',
                  'path_id',
                  'x',
                  'y',
                  'z',
                  'dot_product',
                  'length']
        writer.writerow(header)
        # for each cells
        for cell_id in range(len(cells_info)):
            # for each cytonemes in a cell
            for i in range(len(cells_info[cell_id]['path'])):
                path_id = i + 1  # start id from 1
                length = 0
                # for each voxels in a cytoneme in the cell
                n = len(cells_info[cell_id]['path'][i])
                for j in range(n):
                    writer.writerow([
                        cell_id + 1,  # start id from 1 instead of 0
                        cells_info[cell_id]['body_z'][0],
                        cells_info[cell_id]['body_z'][1],
                        round(cells_info[cell_id]['cyto_to_cell'], 4),
                        path_id,
                        cells_info[cell_id]['path'][i][j][2],
                        cells_info[cell_id]['path'][i][j][1],
                        cells_info[cell_id]['path'][i][j][0],
                        round(cells_info[cell_id]['odds'][i][j], 4),
                        round(length, 4)
                    ])
                    # update path length for next voxel in the cytoneme
                    if j + 1 < n:
                        length += euclidean_dist(cells_info[cell_id]['path'][i][j],
                                                 cells_info[cell_id]['path'][i][j+1],
                                                 distances)
    
    print('Results written in', filename)
    
    return