# -*- coding: utf-8 -*-
import yaml
import os
import tifffile as tif
import numpy as np
import random
import carreno.processing.transforms as tfs

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


def split_dataset(dir):
    """
    Split dataset between training (60%), validation (20%) and test (20%).
    volumes specified in YAML config file.
    Parameters
    ----------
    dir : str
        Path to files to split
    Returns
    -------
    train_data : list
        List of training data
    valid_data : list
        List of validation data
    test : list
        List of evaluation data
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
    vol = list(os.listdir(dir))
    train, valid, test = [[] for i in range(3)]
    
    if not 'validation' in config['TRAINING'] or not 'evaluation' in config['TRAINING']:
        # update config with random volumes for validation or evaluation
        # TODO
        ctrl_vol = elem_with_substrings(vol, ["ctrl"])
        slik_vol = elem_with_substrings(vol, ["slik"])
        pass

    target_vol = config['TRAINING']['validation']
    for v in vol:
        if v.rsplit(".", 1)[0] in target_vol:
            valid.append(v)
    
    test_vol = config['TRAINING']['evaluation']
    for v in vol:
        if v.rsplit(".tif", 1)[0] in test_vol:
            test.append(v)
    
    for v in vol:
        if not v in valid and not v in test:
            train.append(v)
    
    return train, valid, test


def augmentations(shape, is_2D, n_color_ch):
    """
    Get augmentations for training and test data.
    Parameters
    ----------
    shape : [int]
        Shape of data sample.
    is_2D : bool
        If the data is 2D or not
    n_color_ch : int
        Number of color channel for X data.
    """
    train_aug = tfs.Compose(transforms=[
        tfs.Read(),
        tfs.PadResize(shape=shape, mode='reflect'),
        tfs.Sample(shape=shape),
        tfs.Normalize(),
        tfs.Flip(axis=1, p=0.5),
        tfs.Flip(axis=2, p=0.5),
        tfs.Rotate([-30, 30], axes=[1,2], order=1, mode='reflect', p=0.5),
        tfs.Squeeze(axis=0, p=(1 if is_2D else 0)),
        tfs.Stack(axis=-1, n=n_color_ch)
    ])
    
    test_aug = tfs.Compose(transforms=[
        tfs.Read(),
        tfs.PadResize(shape=shape, mode='reflect'),
        tfs.Sample(shape=shape),
        tfs.Standardize(),
        tfs.Squeeze(axis=0, p=(1 if is_2D else 0)),
        tfs.Stack(axis=-1, n=n_color_ch)
    ])
    
    return train_aug, test_aug
    

if __name__ == "__main__":
    cf = get_config()
    
    for k, v in cf.items():
        print(k)
        print(v)
    
    print()
    
    trn, vld, tst = split_dataset(cf['VOLUME']['input'])
    print("TRAINING")
    print(trn)
    print("VALIDATION")
    print(vld)
    print("TESTING")
    print(tst)