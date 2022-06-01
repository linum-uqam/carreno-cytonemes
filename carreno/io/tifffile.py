# -*- coding: utf-8 -*-

import tifffile as tif

def metadata(path):
    """Get Tiff file metadata (only support imagej at the moment)
    Parameters
    ----------
    path : str, Path
        Path to Tiff file
    Returns
    -------
    infos : dict{str}
        File metadata organized in map.
        
        Keys :
        - axe : axe order (example XYZ or ZYX)
        - shape : ndarray shape
        - dtype : ndarray dtype
        - kind : metadata type (example imagej or shaped)
        - axe_dist : distance between instances on each axes
        - axe_unit : distance unit for each axes
    """
    infos = {}
    
    file_info = tif.TiffFile(path)
    series_info = file_info.series[0]
    
    infos["axe"] = series_info.axes
    infos["shape"] = series_info.shape
    infos["dtype"] = series_info.dtype
    infos["kind"] = series_info.kind
    infos["axe_dist"] = [None] * len(series_info.axes)
    infos["axe_unit"] = [None] * len(series_info.axes)
    
    if file_info.is_imagej:
        metadata = file_info.imagej_metadata
        axes_id_info = []
        axes_dist_info = []
        axes_unit_info = []
        
        pattern = "Scaling|Distance|"
        for line in metadata["Info"].split("\n"):
            if line[:len(pattern)] == pattern:
                line_end = line[len(pattern):].split(" = ")
                param_upper = line_end[0].upper()
                if "ID" in param_upper:
                    axes_id_info.append(line_end)
                elif "VALUE" in param_upper:
                    axes_dist_info.append(line_end)
                elif "UNIT" in param_upper:
                    axes_unit_info.append(line_end)
        
        def sort_param(x):
            return sorted(x, key=lambda i: i[0])
        
        # they should already be sorted, but just to make sure
        axes_id_info = sort_param(axes_id_info)
        axes_dist_info = sort_param(axes_dist_info)
        axes_unit_info = sort_param(axes_unit_info)
        
        for i in range(len(axes_id_info)):
            ax = axes_id_info[i][1]
            ax_idx = series_info.axes.index(ax)
            infos["axe_dist"][ax_idx] = axes_dist_info[i][1]
            infos["axe_unit"][ax_idx] = axes_unit_info[i][1]
    
    return infos


def write_imagej_tif(path, x, axes="ZYX", distances=[1, 1, 1], units=['µm', 'µm', 'µm']):
    """Write a tif image with a minimum of useful metadata (update depending on needs)
    Parameters
    ----------
    path : str, Path
        Path to write ndarray to
    x : ndarray
        ndarray to save
    axes : str
        Axes order ("XY", "YX")
    distances : [float]
        distances between instances for each axes
    units : [str]
        units of distance used for each axes distance
    """
    # resolution parameter with imwrite doesn't seem to work with imagej format
    # so we partially recreate the Info metadata for distance information
    dist_prefix = "Scaling|Distance|"
    dist_unit = dist_prefix + "DefaultUnitFormat #"
    dist_id = dist_prefix + "Id #"
    dist_val = dist_prefix + "Value #"
    sep = " = "
    info_meta = ""
    
    def create_line(name, idx, sep, val):
        return name + str(idx+1) + sep + val + "\n"
    
    for i in range(len(units)):
        info_meta += create_line(dist_unit, i, sep, units[i])
    
    for i in range(len(axes)):
        info_meta += create_line(dist_id, i, sep, axes[i])
    
    for i in range(len(distances)):
        info_meta += create_line(dist_val, i, sep, str(distances[i]))
    
    tif.imwrite(path, x, imagej=True,
                metadata={
                    'axes': axes,
                    "Info": info_meta
                })