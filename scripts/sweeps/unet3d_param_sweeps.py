# -*- coding: utf-8 -*-
import wandb
import os

# local imports
import utils

os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]= "20"

sweep_configs = {
    0 : {
        'method': 'bayes',
        'name':   'sweep',
        'project': 'unet3d_lr2',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'ndim':  {'value': 3},
            'shape': {'value': [32, 48, 48]},
            'bsize': {'value': 8},
            'dupe':  {'value': 1},
            'lr':    {'min': 0.00001, 'max': 0.1}
        }
    },
    1 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'unet3d_size3',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'ndim':  {'value': 3},
            'shape': {'values': [[16, 128, 128], [16, 112, 112], [16, 96, 96]]},
            'bsize': {'values': [2, 3, 4]},
            'dupe':  {'value': 8}
        }
    }
}


def main():
    current_test = 1
    sweep_config = sweep_configs[current_test]
    sweep_id = wandb.sweep(sweep_config)
    sweeper = utils.UNetSweeper(sweep_config,wandb_artifact=False)
    wandb.agent(sweep_id, function=sweeper.sweep)
    

if __name__ == "__main__":
    main()