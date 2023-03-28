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


def split_patches(dir):
    """
    Split patches in dataset based on validation and test
    volumes specified in YAML config file.
    Parameters
    ----------
    dir : str
        Path to files to split
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
    patches = list(os.listdir(dir))
    valid_patch = elem_with_substrings(patches, config['TRAINING']['validation'])
    test_patch  = elem_with_substrings(patches, config['TRAINING']['evaluation'])
    train_patch = []
    for patch in patches:
        if not patch in valid_patch and not patch in test_patch:
            train_patch.append(patch)
    
    return train_patch, valid_patch, test_patch


def main():
    cf = get_config()
    for k, v in cf.items():
        print(k)
        print(v)
    
    """
    print()
    trn, vld, tst = split_patches(cf['PATCH']['input'])
    print("TRAINING")
    print(trn)
    print("VALIDATION")
    print(vld)
    print("TESTING")
    print(tst)
    """
    
if __name__ == "__main__":
    main()