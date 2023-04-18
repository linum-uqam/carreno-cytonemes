# -*- coding: utf-8 -*-
import wandb

# local imports
import utils

sweep_configs = {
    1 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'unet2d_lr_batchsize',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'lr':    {'values': [0.01, 0.001, 0.0001]},
            'bsize': {'values': [16, 32, 48]}
        }
    },
    2 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'unet2d_scaler',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'lr':     {'value': 0.001},
            'bsize':  {'value': 32},
            'scaler': {'values': ['norm', 'std']},
            'label':  {'values': ['soft', 'hard']}
        }
    },
    3 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'unet2d_batch_norm_order',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'lr':     {'value': 0.001},
            'bsize':  {'value': 32},
            'scaler': {'value': 'std'},
            'label':  {'value': 'soft'},
            'order':  {'values': ['before', 'after']}
        }
    },
    4 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'unet2d_loss_act',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'lr':     {'value': 0.001},
            'bsize':  {'value': 32},
            'scaler': {'value': 'std'},
            'label':  {'value': 'soft'},
            'order':  {'value': 'after'},
            'act':    {'values': ['relu', 'elu', 'gelu']},
            'topact': {'values': ['relu', 'softmax']},
            'loss':   {'values': ['dice', 'cedice', 'dicecldice', 'adawing', 'cldiceadawing']}
        }
    },
    5 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'unet2d_dropout',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'lr':      {'value': 0.001},
            'bsize':   {'value': 32},
            'scaler':  {'value': 'std'},
            'label':   {'value': 'soft'},
            'order':   {'value': 'after'},
            'act':     {'value': 'relu'},
            'topact':  {'value': 'relu'},
            'loss':    {'value': 'cldiceadawing'},
            'dropout': {'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
        }    
    },
    6 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'unet2d_backbone_color',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'lr':     {'value': 0.001},
            'size':   {'value': 32},
            'scaler': {'value': 'std'},
            'label':  {'value': 'soft'},
            'order':  {'value': 'after'},
            'act':    {'value': 'relu'},
            'topact': {'value': 'relu'},
            'loss':   {'value': 'cldiceadawing'},
            'backbone': {'value': 'vgg16'},
            'pretrn': {'value': True},
            'ncolor': {'values': [1, 3]}
        }
    }
}


def main():
    current_test = 6
    sweep_config = sweep_configs[current_test]
    sweep_id = wandb.sweep(sweep_config)
    sweeper = utils.Sweeper(sweep_config)
    wandb.agent(sweep_id, function=sweeper.sweep)
    

if __name__ == "__main__":
    main()