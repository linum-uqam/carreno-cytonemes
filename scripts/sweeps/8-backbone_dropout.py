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
    'project': 'unet2d_backbone_color',
    'metric': {
        'goal': 'maximize',
        'name': 'val_dicecldice'
    },
    'parameters': {
        'lr':      {'value': 0.001},
        'size':    {'value': 64},
        'scaler':  {'value': 'norm'},
        'label':   {'value': 'soft'},
        'order':   {'value': 'after'},
        'ncolor':  {'value': 3},
        'act':     {'value': "relu"},
        'loss':    {'value': "cldice"},
        'dropout': {'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
    }
}


def main():
    sweep_id = wandb.sweep(sweep_config)
    sweeper = utils.Sweeper(sweep_config)
    wandb.agent(sweep_id, function=sweeper.sweep)


if __name__ == "__main__":
    main()