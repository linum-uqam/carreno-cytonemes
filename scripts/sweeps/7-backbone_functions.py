# -*- coding: utf-8 -*-
import tensorflow as tf
import wandb
import os
import numpy as np
from pathlib import Path

# local imports
import utils
from carreno.nn.unet import UNet
import carreno.nn.metrics as mtc
from carreno.nn.unet import encoder_trainable, switch_top
from carreno.nn.generators import Generator

sweep_config = {
    'method': 'grid',
    'name':   'sweep',
    'project': 'unet2d_vgg16_fonctions',
    'metric': {
        'goal': 'maximize',
        'name': 'val_dicecldice'
    },
    'parameters': {
        'lr':     {'value': 0.001},
        'size':   {'value': 64},
        'scaler': {'value': 'norm'},
        'label':  {'value': 'soft'},
        'order':  {'value': 'after'},
        'ncolor': {'value': 3},
        'act':    {'values': ["relu", "leaky_relu", "elu", "gelu"]},
        'loss':   {'values': ["dice", "bce_dice", "cldice"]}
    }
}


def main():
    sweep_id = wandb.sweep(sweep_config)
    sweeper = utils.Sweeper(sweep_config)
    wandb.agent(sweep_id, function=sweeper.sweep)


if __name__ == "__main__":
    main()