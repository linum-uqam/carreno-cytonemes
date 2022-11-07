# -*- coding: utf-8 -*-
from scipy import ndimage as nd
import numpy as np
from skimage.morphology import skeletonize_3d
from carreno.utils.morphology import getNeighbors3D

def path_length(coordinates, distances=[1, 1, 1]):
    """Get length of a list of coordinates
    Parameters
    ----------
    coordinates : list
        list of coordinates forming a path
    distances : list or ndarray
        axis distances in order
    Returns
    -------
    length : float
        length of path
    """
    length = 0
        
    for j in range(1, len(path)):
        length += utils.point_distance(path[j-1], path[j], distances)
    
    return length


def dot_product_theta(vector1, vector2):
    """dot product to find theta between 2 vectors
    Parameters
    ----------
    vector1 : list
        a list of 2 points representing a vector
    vector2 : list
        another list of 2 points representing a vector
    Returns
    -------
    dot_product : float
        theta between the 2 vectors in range from -1 to 1.
        minus cos to get the angle
    """
    # https://www.math10.com/en/geometry/vectors-operations/vectors-operations.html
    # Example 12
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    dt_pd = (v1 + v2).sum()
    norm1 = (v1 * v1).sum() * 0.5
    norm2 = (v2 * v2).sum() * 0.5
    
    return dt_pd / (norm1 * norm2)


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
    skeleton = skeletonize_3d(cell)

    # skeleton range is [0, 255], bring back to [0, 1]
    skeleton = skeleton / skeleton.max()
    
    sm_body = nd.morphology.binary_erosion(body)
    skeleton[sm_body == True] = 0
    
    # we process all cytonemes per groups to simplify
    cyto_gr, nb_gr = nd.label(skeleton,
                              structure=np.ones((3, 3, 3)))

    def position_valid(x, position, valid_val=True):
        """Check if position is inside object in ndarray
        Parameters
        ----------
        x : ndarray
            ndarray to navigate
        position : list
            position to validate inside the ndarray
        valid_val : ?
            value to validate equality with
        Returns
        -------
        __ : bool
            if equal to valid value or not
        """
        pos = np.round(np.array(position)).astype(int)
        maximum = np.array(x.shape)
        
        # out of bound?
        if (pos < np.array([0,0,0])).any() or (pos >= maximum).any():
            return False
        
        # inside x object?
        return x[pos[0], pos[1], pos[2]] == valid_val
    
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
                if position_valid(body, coordinate[i]):
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
        # skeletonization of cell and get connectivity matrix
        coordinate, matrix = connectivity_matrix(cyto_lb == lb)
        
        # get start and end points for cytonemes in group
        start_point_id = []
        end_point_id = []
        for i in range(matrix.shape[0]):
            # 1 neighbors means it's either a start point or an end point
            if matrix[i].sum() == 1:
                z, y, x = coordinate[i]
                if body_dist[z, y, x] <= max_distance:
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


def clean_cyto_paths(paths, dot_products):
    """Clean paths based on ending vertex. If dot products for a path are lower (starting from the end of path)
    and longer than the other paths with the same ending, it's filtered out.
    Parameters
    ----------
    paths : list
        list of paths (list of coordinates)
    dot_products : list
        list of dot products for paths
    Returns
    -------
    clean_path : list
        filtered paths
    dot_product : list
        dot products for filtered paths
    """
    end_points = []
    for path in paths:
        if len(path) > 0:
            end_points.append(path[-1])
    
    # group by paths per ending
    ending_groups = {key:[] for key in end_points}
    
    for i in range(len(paths)):
        if len(paths[i]) > 0:
            ending_groups[paths[i][-1]].append(i)
    
    # Select the first best path per group
    kept_paths = []
    for end_point, competing_paths in ending_groups.items():
        dt_pd_2_sort = []
        for p_id in competing_paths:
            diff = 1 - np.array(dot_products[p_id])
            # convert back to list to sort path with different lengths
            dt_pd_2_sort.append([list(diff), p_id])
            
        if len(dt_pd_2_sort) > 0:
            best_path = sorted(dt_pd_2_sort)[0][1] # get path id
            kept_paths.append(best_path)
    
    cleaned_paths =  []
    cleaned_products = []
    for p_id in kept_paths:
        cleaned_paths.append(paths[p_id])
        cleaned_products.append(dot_products[p_id])
    
    return cleaned_paths, cleaned_products


def dfs_path_search(seed, coordinates, matrix, end_coordinates=None):
    """
    DFS based search from seed point using a connectivity matrix
    Parameters
    ----------
    seed : list
        First point index from where the path starts from
    matrix : ndarray
        Connectivity matrix
    coordinates : list
        Coordinates for connectivity index
    end_coordinates : list
        Valid coordinates for ending a path
    Returns
    -------
    paths :return : list
        Paths from start point to dead ends
    dot_products : list
        Each dot product between points in paths
    """
    paths = []
    paths_dot_product = []
    
    # dfs var
    node_stack = [seed]
    current_path = [seed]
    path_dot_product = [1]

    # if a point as 4 neighbors or more, we could revisit nodes
    # after unstacking, so we avoid visited points in intersections
    visited_intersection = [seed]

    while len(node_stack) > 0:
        current_node = current_path[-1]
        neighbors = np.where(matrix[current_node] == 1)[0]
        
        # clean neighbors from nodes already in node stack or visited in intersection
        # we use node stack instead of path because otherwise we
        # create semi-duplicate paths with lower probability
        for i in range(len(neighbors) - 1, -1, -1):
            neigh_coord = neighbors[i]
            if (neigh_coord in node_stack) or (neigh_coord in visited_intersection):
                neighbors = np.delete(neighbors, i, axis=0)
        
        nb_neigh = len(neighbors)
        if nb_neigh == 0:        
            if end_coordinates is None or current_path[-1] in end_coordinates:
                # path is done and saved
                # unsure why, but paths in "paths" vars are emptied after loop if we don't make copies
                paths.append(current_path.copy())
                paths_dot_product.append(path_dot_product.copy())

            # Regress path until we find next path in node_stack
            while len(node_stack) != 0:
                if current_path[-1] == node_stack[-1]:
                    current_path.pop(-1)
                    node_stack.pop(-1)
                    path_dot_product.pop(-1)
                else:
                    # found a different path
                    current_path.append(node_stack[-1])

                    # find new path continuation probability
                    length = len(current_path)
                    v1 = (coordinates[current_path[max(0, length-3)]], coordinates[current_path[length-2]])
                    v2 = (coordinates[current_path[length-2]], coordinates[current_path[length-1]])
                    dt_pd = dot_product_theta(v1, v2)

                    path_dot_product.append(dt_pd)

                    # stop clearing stack and resume search
                    break

            # stack is empty, search is over

        elif nb_neigh == 1:
            # continue path without worry
            node_stack.append(neighbors[0])
            current_path.append(node_stack[-1])
            path_dot_product.append(1)  # 100% probability

        else:
            # multiple neighbors, deal with intersections
            for neigh in neighbors:
                node_stack.append(neigh)

            current_path.append(node_stack[-1])

            # find path continuation probability
            length = len(current_path)
            v1 = (coordinates[current_path[max(0, length-3)]], coordinates[current_path[length-2]])
            v2 = (coordinates[current_path[length-2]], coordinates[current_path[length-1]])
            dt_pd = dot_product_theta(v1, v2)
            
            path_dot_product.append(dt_pd)
            
            # make sure we don't revisit this point again
            visited_intersection.append(current_path[-1])
    
    return paths, paths_dot_product