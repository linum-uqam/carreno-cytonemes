# -*- coding: utf-8 -*-
import csv
from carreno.utils.util import euclidean_dist


def cells_info_csv(filename, cells_info, distances):
    """Write paths info for one cell into a csv file
    Parameters
    ----------
    filename : path, str
        Location to save csv file
    cells_info : [map]
        Contains all the metric we want to save
        Expected map keys and values:
        - 'body_z' : [start_idx, end_idx]
        - 'path' : [cytoneme_path]
        - 'odds': [path_probability]
    distances : list, ndarray
        Axis distances in order for calculating length between coordinates
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