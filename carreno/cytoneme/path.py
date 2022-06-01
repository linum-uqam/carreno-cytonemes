from scipy import ndimage as nd
import numpy as np


def connectivity_matrix(x):
    """
    Creates a connectivity matrix based on a volume.
    Connectivity is 26.
    :param x: volume (ndarray)
    :return: coordinates list for matrix index (list)
    :return: 2D connectivity matrix (ndarray)
    """
    # label each value
    coords = np.stack(np.where(x >= 1), axis=1)
    nb_lb = len(coords)

    nodes = {}
    for i in range(nb_lb):
        nodes[tuple(coords[i])] = i

    # init connectivity matrix
    con_matrix = np.zeros((nb_lb, nb_lb))
    
    # index coord in matrix
    coord_list = []
        
    for coord, lb in nodes.items():
        coord_list.append(coord)
        neighbors = getNeighbors3D(coord, x)

        # put connections in matrix
        for neigh_coord in neighbors:
            # lb of neighbor
            neigh_lb = nodes[tuple(neigh_coord)]
            con_matrix[lb, neigh_lb] = 1
    
    return coord_list, con_matrix


def path_id_2_coordinate(path, coordinate):
    """Takes a path of coordinates id and convert to a path of coordinates
    Parameters
    ----------
    path : list
        list of coordinates id
    coordinates : list
        list of coordinates for id convertion
    Returns
    -------
    coord_path : list
        list of coordinates
    """
    coord_path = []
    
    for coord_id in path:
        coord_path.append(coordinate[coord_id])
    
    return coord_path


def skeletonized_cell_paths(cell, body):
    """Gets cytonemes path for a single cell using its body edge as the starting point of paths.
    Parameters
    ----------
    cell : ndarray
        mask of cell
    body : ndarray
        mask of cell body
    Returns
    -------
    paths: list
        cytonemes paths
    dot_products: list
        cytonemes dot product between each coordinates where theres an intersection
    """
    paths = []
    dot_products = []
    
    # skeletonize cell for clear pathing
    skeleton = morphology.skeletonize_3d(cell)

    # skeleton range is [0, 255], bring back to [0, 1]
    skeleton = skeleton / skeleton.max()
    
    sm_body = nd.morphology.binary_erosion(body)
    skeleton[sm_body == True] = 0
    
    # we process all cytonemes per groups to simplify
    cyto_gr, nb_gr = nd.label(skeleton,
                              structure=np.ones((3, 3, 3)))
    
    for j in range(1, nb_gr + 1):
        gr = cyto_gr == j
        
        # skeletonization of cell and get connectivity matrix
        coordinate, matrix = connectivity_matrix(gr)
        
        # get start and end points for cytonemes in group
        start_point_id = []
        end_point_id = []
        for i in range(matrix.shape[0]):
            # 1 neighbors means it's either a start point or an end point
            if matrix[i].sum() == 1:
                if __position_inside(coordinate[i], body):
                    start_point_id.append(i)
                else:
                    end_point_id.append(i)
        
        # get cytonemes path in current group
        for st_p in start_point_id:
            tmp_id_path, tmp_dot = dfs_path_search(st_p,
                                                   coordinate,
                                                   matrix,
                                                   end_point_id)
            
            # convert paths of ids to paths of coordinates
            coordinate_path = []
            for p in tmp_id_path:
                coordinate_path.append(path_id_2_coordinate(p, coordinate))
            
            # keep paths coordinates and probabilities
            paths += coordinate_path
            dot_products += tmp_dot
    
    return paths, dot_products


def skeletonized_cyto_paths(body, cyto_lb, association, max_distance=5):
    """Matches cytonemes with the nearest body.
    Parameters
    ----------
    body : ndarray
        cell body mask
    cyto_label : ndarray
        labelled cytonemes
    association : list
        list of label to consider
    max_distance : float
        maximum distance from a cell body to be considered as the cytoneme start
    Returns
    -------
    paths: list
        cytonemes paths
    dot_products: list
        cytonemes dot product between each coordinates where theres an intersection
    """
    paths = []
    dot_products = []
    body_dist = nd.distance_transform_edt(body == 0)
    
    for lb in association:
        c = cyto_lb == lb
        
        # skeletonization of cell and get connectivity matrix
        coordinate, matrix = connectivity_matrix(c)
        
        # get start and end points for cytonemes in group
        start_point_id = []
        end_point_id = []
        for i in range(matrix.shape[0]):
            # 1 neighbors means it's either a start point or an end point
            if matrix[i].sum() == 1:
                d, r, c = coordinate[i]
                if body_dist[d, c, r] <= max_distance:
                    start_point_id.append(i)
                else:
                    end_point_id.append(i)
        
        # get cytonemes path in current group
        for st_p in start_point_id:
            tmp_id_path, tmp_dot = dfs_path_search(st_p,
                                                   coordinate,
                                                   matrix,
                                                   end_point_id)
            
            # convert paths of ids to paths of coordinates
            coordinate_path = []
            for p in tmp_id_path:
                coordinate_path.append(path_id_2_coordinate(p, coordinate))
            
            # keep paths coordinates and probabilities
            paths += coordinate_path
            dot_products += tmp_dot
    
    return paths, dot_products