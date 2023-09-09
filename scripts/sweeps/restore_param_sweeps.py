# -*- coding: utf-8 -*-
import wandb

# local imports
import utils

sweep_configs = {
    0 : {
        'method': 'bayes',
        'name':   'sweep',
        'project': 'threshold_denoise2',
        'metric': {
            'goal': 'maximize',
            'name': 'noise'
        },
        'parameters': {
            'psf':       {'values': [False, True]},
            'iteration': {'min': 5, 'max': 60},
            'nlm_size':  {'min': 1, 'max': 10},
            'nlm_dist':  {'min': 1, 'max': 10},
        }
    },
    1 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'threshold_restore_optim',
        'metric': {
            'goal': 'maximize',
            'name': 'noise'
        },
        'parameters': {
            'psf':       {'value': True},
            'iteration': {'value': 60},
            'nlm_size':  {'value': 0},
            'nlm_dist':  {'value': 0},
            'r':         {'value': 50},
            'amount':    {'value': 8},
            'md_ax0':    {'value': 3},
            'md_ax12':   {'value': 5},
            'btw_h':     {'value': 0.5},
            'btw_l':     {'value': 0.95},
            'btw_s':     {'value': 0.09},
        }
    }
}


def main():
    current_test = 1
    sweep_config = sweep_configs[current_test]
    sweep_id = wandb.sweep(sweep_config)
    sweeper = utils.RestoreSweeper(sweep_config)
    wandb.agent(sweep_id, function=sweeper.sweep)
    

if __name__ == "__main__":
    main()