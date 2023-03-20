# -*- coding: utf-8 -*-
import yaml
import os
import tifffile as tif
import numpy as np
import random

config_path = 'config.yml'

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

# TODO maybe save volumes split to avoid training on validation or test data later on in another script
#def modify_split(valid, test):
#    """
#    Update dataset split in configuration yaml file
#    Parameters
#    ----------
#    valid : list
#        list of validation volumes
#    test : list
#        list of test volumes
#    Returns
#    -------
#    None
#    """
#    with open('config.yml', 'r') as file:
#        data_split = {
#            "TRAINING":{
#                "split":{
#                    "valid": valid,
#                    "test":  test
#                }
#            }
#        }
#        yaml.dump(data_split, file, default_flow_style=False)
#    return


def split_volumes(vtype, valid, test, shuffle=False):
    """
    Split volumes in dataset between training, validation and test.
    Training ratio is whatever that's left.
    Parameters
    ----------
    vtype : str
        Volume type in config['VOLUME'] for parsing name
    valid : float
        Validation data ratio over dataset
    test : float
        Test data ratio over dataset
    shuffle : bool
        Shuffle data
    Returns
    -------
    train_data : [Path]
        List of volumes for training
    valid_data : [Path]
        List of volumes for validation
    test_data : [Path]
        List of volumes for testing
    """
    config = get_config()
    
    # list ctrl and slik volumes
    ctrl = []
    slik = []
    for f in os.listdir(config['VOLUME'][vtype]):
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

    # write split into yaml file TODO

    return train_data, valid_data, test_data


def split_patches(ptype, pouts, train, valid, test):
    """
    Split patches in dataset based on training, validation and test volumes.
    Parameters
    ----------
    ptype : str
        Patch type in config['PATCH'] for parsing name
    pouts : [str]
        Patch types in config['PATCH'] for output paths
    train : [Path]
        Training volume paths
    valid : [Path]
        Validation volume paths
    test : [Path]
        Testing volume paths
    Returns
    -------
    train_data : dict
        Dict of patches for training
    valid_data : dict
        Dict of patches for validation
    test_data : dict
        Dict of patches for testing
    """
    config = get_config()
    
    # rm file extensions
    data = [train, valid, test]
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
    patches = list(os.listdir(config['PATCH'][ptype]))
    patch_set = []
    for vol_set in data:
        patch_set.append(elem_with_substrings(patches, vol_set))

    def add_dir_to_files(dir, files):
        return [os.path.join(dir, f) for f in files]

    # add parent folders for a working file path
    rtn_value = []
    for patches_for_ctg in patch_set:
        patch_dict = {}
        for pout in pouts:
            patch_dict[pout] = add_dir_to_files(config['PATCH'][pout], patches_for_ctg)
        rtn_value.append(patch_dict)

    return rtn_value


def split_dataset(vtype, pouts, valid, test, shuffle=False):
    """
    Split patches between training, validation and test.
    Training ratio is whatever that's left.
    Parameters
    ----------
    vtype : str
        Volume type in config['VOLUME'] for parsing name
    pouts : [str]
        Patch types in config['PATCH'] for output paths
    valid : float
        Validation data ratio over dataset
    test : float
        Test data ratio over dataset
    shuffle : bool
        Shuffle data
    Returns
    -------
    train_data : dict
        Dict of patches for training
    valid_data : dict
        Dict of patches for validation
    test_data : dict
        Dict of patches for testing
    """
    trn, vld, tst = split_volumes(vtype=vtype,
                                  valid=valid,
                                  test=test,
                                  shuffle=shuffle)
    return split_patches(ptype=vtype,
                         pouts=pouts,
                         train=trn,
                         valid=vld,
                         test=tst)


def main():
    cf = get_config()
    
    for k, v in cf.items():
        print(k)
        print(v)
    
    print()


if __name__ == "__main__":
    main()