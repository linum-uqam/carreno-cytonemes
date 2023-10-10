# -*- coding: utf-8 -*-
import wandb

# local imports
import utils

sweep_configs = {
    0 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'threshold_psf_nlm',
        'metric': {
            'goal': 'maximize',
            'name': 'dice'
        },
        'parameters': {
            'ro':        {'value': 0.5},
            'rc':        {'value': 0.5},
            'md_ax0':    {'value': 3},
            'md_ax12':   {'value': 5},
            'btw_h':     {'value': 0.5},
            'btw_l':     {'value': 0.95},
            'btw_s':     {'value': 0.09},
            'r':         {'value': 50},
            'amount':    {'value': 10},
            'psf':       {'values': [False, True]},
            'iteration': {'min': 5, 'max': 60},
            'nlm_size':  {'min': 1, 'max': 10},
            'nlm_dist':  {'min': 1, 'max': 10},
        }
    },
    1 : {
        'method': 'bayes',
        'name':   'sweep',
        'project': 'threshold_w_butterworth2',
        'metric': {
            'goal': 'maximize',
            'name': 'dice'
        },
        'parameters': {
            'r':         {'value': 50},
            'amount':    {'value': 7},
            'ro':        {'value': 1},
            'rc':        {'value': 1.5},
            'md_ax0':    {'value': 5},
            'md_ax12':   {'value': 5},
            'btw_h':     {'min': 0.0, 'max': 0.5},
            'btw_l':     {'min': 0.2, 'max': 1.0},
            'btw_s':     {'min': -0.1, 'max': 0.1},
        }
    },
    2 : {
        'method': 'bayes',
        'name':   'sweep',
        'project': 'threshold_affine_filter2',
        'metric': {
            'goal': 'maximize',
            'name': 'dice'
        },
        'parameters': {
            'ro':        {'value': 0.5},
            'rc':        {'value': 0.5},
            'md_ax0':    {'value': 3},
            'md_ax12':   {'value': 5},
            'btw_h':     {'value': 0.5},
            'btw_l':     {'value': 0.95},
            'btw_s':     {'value': 0.09},
            'r':         {'min': 1.0, 'max': 50.0},
            'amount':    {'min': 0.0, 'max': 10.0}
        }
    },
    3 : {
        'method': 'bayes',
        'name':   'sweep',
        'project': 'threshold_wo_butterworth2',
        'metric': {
            'goal': 'maximize',
            'name': 'dice'
        },
        'parameters': {
            'md_ax0':    {'value': 3},
            'md_ax12':   {'value': 5},
            'btw_h':     {'value': 0.5},
            'btw_l':     {'value': 0.95},
            'btw_s':     {'value': 0.09},
            'md_ax0':    {'min': 1, 'max': 5},
            'md_ax12':   {'min': 1, 'max': 5},
            'ro':        {'min': 0.25, 'max': 2.0},
            'rc':        {'min': 0.25, 'max': 2.0}
        }
    },
    4 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'threshold_optim3',
        'metric': {
            'goal': 'maximize',
            'name': 'dice'
        },
        'parameters': {
            'ro':        {'value': 0.5},
            'rc':        {'value': 0.5},
            'md_ax0':    {'value': 3},
            'md_ax12':   {'value': 5},
            'btw_h':     {'value': 0.5},
            'btw_l':     {'value': 0.95},
            'btw_s':     {'value': 0.09},
            'r':         {'value': 50},
            'amount':    {'value': 10}
        }
    }
}


def main():
    current_test = 0
    sweep_config = sweep_configs[current_test]
    sweep_id = wandb.sweep(sweep_config)
    sweeper = utils.ThresholdSweeper(sweep_config)
    wandb.agent(sweep_id, function=sweeper.sweep)
    

if __name__ == "__main__":
    main()