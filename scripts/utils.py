# -*- coding: utf-8 -*-
import yaml
import os
import tifffile as tif
import numpy as np
import random

def get_config():
    """
    Get configutations from configuration yaml file
    Returns
    -------
    infos: dict
        dictionnary with config infos
    """
    infos = {}
    with open('config.yml', 'r') as file:
        infos = yaml.safe_load(file)
    return infos


def get_volumes_slices(paths):
    """
    Get all slice index for all the volumes in list of paths
    Parameters
    ----------
    paths : [str]
        Paths to volumes to slice up
    Returns
    -------
    slices : [[str, int]]
        list of list containing volume names and slice indexes
    """
    slices = []

    for path in paths:
        tif_file = tif.TiffFile(path)
        for i in range(len(tif_file.pages)):
            slices.append([path, i])
    
    return slices


def split_dataset(valid, test, shuffle=False):
    """
    Split dataset in training, validation and test. Training is whatever that's left.
    Parameters
    ----------
    valid : float
        validation data ratio over dataset
    test : float
        test data ratio over dataset
    shuffle : bool
        shuffle data
    Returns
    -------
    train_data : dict
        dict of patches for training
    valid_data : dict
        dict of patches for validation
    test_data : dict
        dict of patches for testing
    """
    config = get_config()
    
    # list ctrl and slik volumes
    ctrl = []
    slik = []
    for f in os.listdir(config['VOLUME']['input']):
        lower_f = f.lower()
        if "ctrl" in lower_f:
            ctrl.append(f)
        elif "slik" in lower_f:
            slik.append(f)
    
    if shuffle:
        random.shuffle(ctrl)
        random.shuffle(slik)

    # find size for each categories
    total_size = len(ctrl) + len(slik)
    test_size =  np.round(total_size * test)
    valid_size = np.round(total_size * valid)
    #train_size = total_size - test_size - valid_size  # ex : 10 - 2 - 2 == 6
    
    def __fl_n_ce_of_half(x):
        half = x / 2
        return [int(i) for i in [np.floor(half), np.ceil(half)]]
    
    # split dataset
    test_data, valid_data, train_data = [[]] * 3

    # test data split
    fl, ce = __fl_n_ce_of_half(test_size)
    test_data = ctrl[-fl:] + slik[-ce:]
    
    # rm test data from data pool
    ctrl = ctrl[:-fl]
    slik = slik[:-ce]

    # validation data split
    fl, ce = __fl_n_ce_of_half(valid_size)
    valid_data = ctrl[-fl:] + slik[-ce:]

    # rm valid data from data pool
    ctrl = ctrl[:-fl]
    slik = slik[:-ce]

    # training data split
    train_data = ctrl + slik
    
    # rm file extensions
    data = [train_data, valid_data, test_data]
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = data[i][j].rsplit('.', 1)[0]
    
    # from here, we established which volume is used for train, validation
    # or test but we need to group up patches for each of them

    def elem_with_substrings(x, substrings):
        """
        Remove list elements without substrings in them.
        Parameters
        ----------
        x : [str]
            list of strings
        substrings : [str]
            substrings list to look for in `x`
        Returns
        -------
        contains : list
            list of strings containing substring
        """
        contains = []

        for elem in x:
            for sbstr in substrings:
                if sbstr in elem:
                    contains.append(elem)

        return contains

    # seperate patches based on split
    patches = list(os.listdir(config['PATCH']['input']))
    train_p = elem_with_substrings(patches, train_data)
    valid_p = elem_with_substrings(patches, valid_data)
    test_p  = elem_with_substrings(patches, test_data)

    def add_dir_to_files(dir, files):
        return [os.path.join(dir, f) for f in files]

    # add parent folders for a working file path
    patch_ctgs = [train_p, valid_p, test_p]
    rtn_value = []
    for patches_for_ctg in patch_ctgs:
        rtn_value.append({
            'x':  add_dir_to_files(config['PATCH']['input'],       patches_for_ctg),
            'y':  add_dir_to_files(config['PATCH']['target'],      patches_for_ctg),
            'sy': add_dir_to_files(config['PATCH']['soft_target'], patches_for_ctg),
            'w':  add_dir_to_files(config['PATCH']['weight'],      patches_for_ctg),
            'sw': add_dir_to_files(config['PATCH']['soft_weight'], patches_for_ctg),
        })

    return rtn_value


def main():
    cf = get_config()
    
    for k, v in cf.items():
        print(k)
        print(v)
    
    print()


if __name__ == "__main__":
    main()