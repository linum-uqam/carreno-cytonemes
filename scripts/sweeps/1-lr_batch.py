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
    'project': 'unet2d_lr_batchsize',
    'metric': {
        'goal': 'maximize',
        'name': 'val_dicecldice'
    },
    'parameters': {
        'lr':   {'values': [0.01, 0.005, 0.001, 0.0005, 0.0001]},
        'bsize': {'values': [32, 64, 96]}
    }
}


def main():
    sweep_id = wandb.sweep(sweep_config)
    sweeper = utils.Sweeper(sweep_config)
    wandb.agent(sweep_id, function=sweeper.sweep)
    

if __name__ == "__main__":
    main()