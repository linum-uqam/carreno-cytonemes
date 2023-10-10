# -*- coding: utf-8 -*-
import yaml
import os
import tifffile as tif
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rc('font', size=18)

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


def augmentations(shape, norm_or_std, is_2D, n_color_ch):
    """
    Get augmentations for training and test data.
    Parameters
    ----------
    shape : [int]
        Shape of data sample.
    norm_or_std : bool
        True for normalisation, False for standardization.
    is_2D : bool
        If the data is 2D or not.
    n_color_ch : int
        Number of color channel for X data.
    Returns
    -------
    train_aug : carreno.processing.transforms.Compose
        List of transformations
    test_aug : carreno.processing.transforms.Compose
        List of transformations
    """
    scaler = tfs.Normalize() if norm_or_std else tfs.Standardize()
    squeeze_p = (1 if is_2D else 0)

    train_aug = tfs.Compose(transforms=[
        tfs.Read(),
        tfs.PadResize(shape=shape, mode='reflect'),
        tfs.Sample(shape=shape),
        scaler,
        tfs.Flip(axis=1, p=0.5),
        tfs.Flip(axis=2, p=0.5),
        tfs.Rotate([-30, 30], axes=[1,2], order=1, mode='reflect', p=0.5),
        tfs.Squeeze(axis=0, p=squeeze_p),
        tfs.Stack(axis=-1, n=n_color_ch)
    ])
    
    test_aug = tfs.Compose(transforms=[
        tfs.Read(),
        tfs.PadResize(shape=shape, mode='reflect'),
        tfs.Sample(shape=shape),
        scaler,
        tfs.Squeeze(axis=0, p=squeeze_p),
        tfs.Stack(axis=-1, n=n_color_ch)
    ])
    
    return train_aug, test_aug


def plot_metrics(path, histories, verbose=0):
    # metrics display (acc, loss, etc.)
    graph_path = path

    def get_color():
        while 1:
            for j in ['b', 'y', 'r', 'g']:
                yield j
    color = get_color()

    for i in range(len(histories)):
        history = histories[i]
        epochs = np.array(history.history['epoch']) + 1
        loss_hist = history.history['loss']
        val_loss_hist = history.history['val_loss']
        plt.plot(epochs, loss_hist, next(color), label='trn {}'.format(i))
        plt.plot(epochs, val_loss_hist, next(color), label='val {}'.format(i))
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_path + "_loss.svg", format="svg")
    plt.savefig(graph_path + "_loss.png")
    plt.show() if verbose else plt.clf()
    
    for i in range(len(histories)):
        history = histories[i]
        epochs = np.array(history.history['epoch']) + 1
        dice_hist = history.history['dice']
        val_dice_hist = history.history['val_dice']
        plt.plot(epochs, dice_hist, next(color), label='trn {}'.format(i))
        plt.plot(epochs, val_dice_hist, next(color), label='val {}'.format(i))
    plt.title('Training and validation F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_path + "_dice.svg", format="svg")
    plt.savefig(graph_path + "_dice.png")
    plt.show() if verbose else plt.clf()

    for i in range(len(histories)):
        history = histories[i]
        epochs = np.array(history.history['epoch']) + 1
        dice_hist = history.history['dicecldice']
        val_dice_hist = history.history['val_dicecldice']
        plt.plot(epochs, dice_hist, next(color), label='trn {}'.format(i))
        plt.plot(epochs, val_dice_hist, next(color), label='val {}'.format(i))
    plt.title('Training and validation DiceClDice')
    plt.xlabel('Epochs')
    plt.ylabel('DiceClDice Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_path + "_dicecldice.svg", format="svg")
    plt.savefig(graph_path + "_dicecldice.png")
    plt.show() if verbose else plt.clf()


def main():
    cf = get_config()
    
    for k, v in cf.items():
        print(k)
        print(v)
    
    """
    print("Patch split:")
    for a, b, c in zip(*split_patches(cf['PATCH']['input'])):
        print(a, b, c)
    """


if __name__ == "__main__":
    main()