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
    'project': 'unet2d_batch_norm_order',
    'metric': {
        'goal': 'maximize',
        'name': 'val_dicecldice'
    },
    'parameters': {
        'lr':     {'value': 0.01},
        'bsize':  {'value': 32},
        'scaler': {'value': 'norm'},
        'label':  {'value': 'soft'},
        'order':  {'values': ['before', 'after']}
    }
}


def main():
    sweep_id = wandb.sweep(sweep_config)
    sweeper = utils.Sweeper(sweep_config)
    wandb.agent(sweep_id, function=sweeper.sweep)


if __name__ == "__main__":
    main()