# -*- coding: utf-8 -*-
import wandb

# local imports
import utils

sweep_configs = {
    0 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'unet2d_class_weight',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'weight': {'values': [False, True]}
        }
    },
    1 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'unet2d_lr_batchsize',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'lr':    {'values': [
                        [0.01, 0, 0.01],
                        [0.001, 0, 0.001],
                        [0.0001, 0, 0.0001]
                     ]},
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
            'lr':     {'value': [0.001, 0, 0.001]},
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
            'lr':     {'value': [0.001, 0, 0.001]},
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
            'lr':     {'value': [0.001, 0, 0.001]},
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
        'method': 'bayes',
        'name':   'sweep',
        'project': 'unet2d_dropout2',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'lr':      {'value': [0.001, 0, 0.001]},
            'bsize':   {'value': 32},
            'scaler':  {'value': 'std'},
            'label':   {'value': 'soft'},
            'order':   {'value': 'after'},
            'act':     {'value': 'relu'},
            'topact':  {'value': 'relu'},
            'loss':    {'value': 'cldiceadawing'},
            'dropout': {'min': 0.0, 'max': 0.5}
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
            'lr':     {'value': [0.001, 0, 0.001]},
            'bsize':  {'value': 32},
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
    },
    7 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'unet2d_bigger_input',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'lr':     {'value': [0.001, 0, 0.001]},
            'bsize':  {'values': [8, 32]},
            'shape':  {'values': [(1, 96, 96), (1, 192, 192)]},
            'scaler': {'value': 'std'},
            'label':  {'value': 'soft'},
            'order':  {'value': 'after'},
            'act':    {'value': 'relu'},
            'topact': {'value': 'relu'},
            'loss':   {'value': 'cldiceadawing'},
            'backbone': {'value': 'vgg16'},
            'pretrn': {'value': True},
            'ncolor': {'value': 3}
        }
    },
    8 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'unet2d_self_trained',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'lr':     {'value': [0.001, 0, 0.001]},
            'bsize':  {'value': 8},
            'shape':  {'value': (1, 192, 192)},
            'scaler': {'value': 'std'},
            'label':  {'value': 'soft'},
            'order':  {'value': 'after'},
            'act':    {'value': 'relu'},
            'topact': {'value': 'relu'},
            'loss':   {'value': 'cldiceadawing'},
            'backbone': {'value': 'vgg16'},
            'pretrn': {'value': True},
            'ncolor': {'value': 3},
        }
    },
    9 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'unet2d_cosine_scheduler',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'bsize':  {'value': 8},
            'shape':  {'value': (1, 192, 192)},
            'scaler': {'value': 'std'},
            'label':  {'value': 'soft'},
            'order':  {'value': 'after'},
            'act':    {'value': 'relu'},
            'topact': {'value': 'relu'},
            'loss':   {'value': 'cldiceadawing'},
            'backbone': {'value': 'vgg16'},
            'pretrn': {'value': True},
            'ncolor': {'value': 3},
            'classw': {'value': [0.37,38.54,4]},
            'lr':     {'values': [
                        [1e-4, 1e-5, 0.1],
                        [1e-4, 5e-4, 0.1],
                        [1e-4, 1e-5, 0.05],
                        [1e-4, 5e-4, 0.05]
                      ]},
            'warmup': {'values': [10, 15, 20]},
            'nepoch': {'values': [30, 40, 50]}
        }
    }
}


def main():
    current_test = 9
    sweep_config = sweep_configs[current_test]
    sweep_id = wandb.sweep(sweep_config)
    sweeper = utils.UNetSweeper(sweep_config,wandb_artifact=False)
    wandb.agent(sweep_id, function=sweeper.sweep)
    

if __name__ == "__main__":
    main()