# -*- coding: utf-8 -*-
import yaml
import os
import tifffile as tif
import numpy as np
import random
import wandb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path
import carreno.processing.transforms as tfs
import carreno.nn.metrics as mtc
from carreno.nn.unet import UNet, encoder_trainable
from carreno.nn.generators import Generator


def get_config(path='config.yml'):
    """
    Get configutations from configuration yaml file
    Parameters
    ----------
    path : Path or str
        Path to config file
    Returns
    -------
    infos: dict
        dictionnary with config infos
    """
    infos = {}
    with open(path, 'r') as file:
        infos = yaml.safe_load(file)
    return infos


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


class Sweeper():
    def __init__(self, config, ndim=2, wandb_artifact=True):
        """
        Setup sweeps for wandb.
        Parameters
        ----------
        config : dict
            Sweeps configuration.
            Refer to Wandb for format https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
        ndim : int
            Number of dimensions for input (2 or 3)
        wandb_artifact : bool
            Save wandb artifacts during training and evaluation.
        """
        self.prj_config = get_config()
        self.swp_config = config
        ndim = 2
        self.wandb_artifact = wandb_artifact
        self.params = {
            'ndim':     ndim,       # 2 or 3
            'shape':    [1 if ndim == 2 else 48, 96, 96],  # data shape
            'depth':    4,          # unet depth
            'nfeat':    64,         # nb feature for first conv layer
            'lr':       0.001,      # learning rate
            'bsize':    64 if ndim == 2 else 3,  # batch size
            'scaler':   'norm',     # normalize or standardize
            'label':    'hard',     # hard or soft input
            'weight':   True,       # use class weights for unbalanced data
            'order':    'before',   # where to put batch norm
            'ncolor':   1,          # color depth for input
            'act':      'relu',     # activation
            'loss':     'dice',     # loss function
            'topact':   'softmax',  # top activation
            'dropout':  0.3,        # dropout rate
            'backbone': 'base',     # base or unet
            'pretrn':   False,      # pretrained on imagenet
            'slftrn':   False,      # self train on unlabeled data
            'dupe':     48          # number of data duplication to fit batch size
        }
    
    def sweep(self):
        wandb.init()

        print("###############")
        print("# UPDATE ATTR #")
        print("###############")
        for param in self.swp_config['parameters'].keys():
            if param in self.params:
                print("-update {} from {} to {}".format(param, self.params[param], getattr(wandb.config, param, None)))
                self.params[param] = getattr(wandb.config, param, self.params[param])
        
        print("###############")
        print("# BUILD MODEL #")
        print("###############")
        
        backbone = None if self.params['backbone'] == 'base' else self.params['backbone']
        self.model = UNet(shape=self.params['shape'][-self.params['ndim']:] + [self.params['ncolor']],
                          n_class=self.prj_config['PREPROCESS']['n_cls'],
                          depth=self.params['depth'],
                          n_feat=self.params['nfeat'],
                          dropout=self.params['dropout'],
                          norm_order=self.params['order'],
                          activation=self.params['act'],
                          top_activation=self.params['topact'],
                          backbone=backbone,
                          pretrained=self.params['pretrn'])

        if not backbone is None:
            encoder_trainable(self.model, False)

        self.model.summary()

        print("###############")
        print("# SET LOADERS #")
        print("###############")

        trn, vld, tst = split_dataset(self.prj_config['VOLUME']['input'])
        fullpath = lambda dir, files : [os.path.join(dir, name) for name in files] * self.params['dupe']
        hard_label = self.params['label'] == 'hard'
        x_train = fullpath(self.prj_config['VOLUME']['input'],  trn)
        y_train = fullpath(self.prj_config['VOLUME']['target' if hard_label else 'soft_target'], trn)
        w_train = fullpath(self.prj_config['VOLUME']['weight' if hard_label else 'soft_weight'], trn) if self.params['weight'] else None
        x_valid = fullpath(self.prj_config['VOLUME']['input'],  vld)
        y_valid = fullpath(self.prj_config['VOLUME']['target'], vld)
        x_test  = fullpath(self.prj_config['VOLUME']['input'],  tst)
        y_test  = fullpath(self.prj_config['VOLUME']['target'], tst)

        print("Training dataset")
        if self.params['weight']:
            print("-nb of instances :", len(x_train), "/", len(y_train), "/", len(w_train))
        else:
            print("-nb of instances :", len(x_train), "/", len(y_train))

        print("Validation dataset")
        print("-nb of instances :", len(x_valid), "/", len(y_valid))

        print("Testing dataset")
        print("-nb of instances :", len(x_test), "/",  len(y_test))

        if self.params['slftrn']:
            files = os.listdir(self.prj_config['VOLUME']['unlabeled'])
            x_slftrn = fullpath(self.prj_config['VOLUME']['unlabeled'],
                                files)
            y_slftrn = fullpath(self.prj_config['VOLUME']['threshold'],
                                files)
            w_slftrn = fullpath(self.prj_config['VOLUME']['threshold_weight'],
                                files)
            print("Unlabeled dataset")
            print("-nb of instances :", len(x_test), "/",  len(y_test))
            

        # setup data augmentation
        train_aug, test_aug = augmentations(shape=self.params['shape'],
                                            norm_or_std=True if self.params['scaler'] == 'norm' else False,
                                            is_2D=self.params['ndim'] == 2,
                                            n_color_ch=self.params['ncolor'])

        # ready up the data generators
        self.train_gen = Generator(x_train,
                                   y_train,
                                   weight=w_train,
                                   size=self.params['bsize'],
                                   augmentation=train_aug,
                                   shuffle=True)
        self.valid_gen = Generator(x_valid,
                                   y_valid,
                                   weight=None,    # not used for validation since it would cause a bias on best saved model
                                   size=self.params['bsize'],
                                   augmentation=test_aug,
                                   shuffle=False)  # make sure all patches fit in epoch
        self.test_gen  = Generator(x_test,
                                   y_test,
                                   weight=None,
                                   size=self.params['bsize'],
                                   augmentation=test_aug,
                                   shuffle=False)
        
        if self.params['slftrn']:
            self.unlabel_gen = Generator(x_slftrn,
                                         y_slftrn,
                                         weight=w_slftrn,
                                         size=self.params['bsize'],
                                         augmentation=train_aug,
                                         shuffle=True)
        
        print("#############")
        print("# CALLBACKS #")
        print("#############")

        # adding "{epoch:02d}-{val_dice:.2f}" to model name is recommended to make it unique
        # but it saves so many different instances of the model...
        param_str = ("-{}" * len(self.swp_config['parameters'])).format(*[getattr(wandb.config, i, None) for i in self.swp_config['parameters']])  # needlessly complicated way to include params in saved model path
        model_name = self.swp_config['project'] + param_str + ".h5"
        model_path = os.path.join(self.prj_config['DIR']['model'], model_name)
        Path(self.prj_config['DIR']['model']).mkdir(parents=True, exist_ok=True)

        # callbacks
        monitor, mode = self.swp_config['metric']['name'], self.swp_config['metric']['goal'][:3]
        early_stop = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                      mode=mode,
                                                      patience=10,
                                                      restore_best_weights=True,
                                                      verbose=1)
        metrics_logger   = wandb.keras.WandbMetricsLogger()
        if self.wandb_artifact:
            checkpoint = wandb.keras.WandbModelCheckpoint(filepath=model_path,
                                                          monitor='val_dicecldice',
                                                          mode='max',
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          verbose=1)
        else:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                            monitor='val_dicecldice',
                                                            mode='max',
                                                            save_best_only=True,
                                                            save_weights_only=False,
                                                            verbose=1)

        total_steps = len(self.train_gen) * self.prj_config['TRAINING']['epoch']
        schedule = tf.keras.optimizers.schedules.CosineDecay(self.params['lr'], decay_steps=total_steps)
        optim = tf.keras.optimizers.Adam(learning_rate=schedule)

        iters = 10
        ndim  = ndim=self.params['ndim']
        pad   = 2
        cls   = slice(1,2)  # only skeletonize cyto (tried, but gives awful result for cldice)
        dice       = mtc.Dice()
        cldice     = mtc.ClDice(    iters=iters, ndim=ndim, mode=pad)
        dicecldice = mtc.DiceClDice(iters=iters, ndim=ndim, mode=pad)
        if self.params['loss'] == "dice":
            loss_fn = dice.loss
        elif self.params['loss'] == "dicecldice":
            loss_fn = dicecldice.loss
        elif self.params['loss'] == "cedice":
            loss_fn = mtc.CeDice().loss
        elif self.params['loss'] == "adawing":
            loss_fn = mtc.AdaptiveWingLoss().loss
        elif self.params['loss'] == "cldiceadawing":
            loss_fn = mtc.ClDiceAdaptiveWingLoss(iters=iters, ndim=ndim, mode=pad).loss
        else:
            raise NotImplementedError

        def compile_n_fit(train_gen, valid_gen, optimizer, initial_epoch=0, log=True):
            self.model.compile(optimizer=optimizer,
                               loss=loss_fn,
                               metrics=[dice.coefficient, cldice.coefficient, dicecldice.coefficient],
                               sample_weight_mode="temporal", run_eagerly=True)
            
            callbacks = [early_stop] + ([checkpoint, metrics_logger] if log else [])
            
            return self.model.fit(train_gen,
                                  validation_data=valid_gen,
                                  steps_per_epoch=len(train_gen),
                                  validation_steps=len(valid_gen),
                                  batch_size=self.params['bsize'],
                                  epochs=self.prj_config['TRAINING']['epoch'],
                                  initial_epoch=initial_epoch,
                                  callbacks=callbacks,
                                  verbose=1)

        if self.params['slftrn']:
            print("##############")
            print("# SELF-TRAIN #")
            print("##############")
            other_schedule = tf.keras.optimizers.schedules.CosineDecay(self.params['lr'], decay_steps=total_steps)
            other_optim    = tf.keras.optimizers.Adam(learning_rate=schedule)
            compile_n_fit(self.unlabel_gen,
                          self.valid_gen,
                          optimizer=other_optim,
                          log=False)

        print("############")
        print("# TRAINING #")
        print("############")

        hist = compile_n_fit(self.train_gen,
                             self.valid_gen,
                             optimizer=optim)
        
        if not backbone is None:
            # train encoder
            encoder_trainable(self.model, True)
            compile_n_fit(self.train_gen,
                          self.valid_gen,
                          optimizer=optim,
                          initial_epoch=len(hist.history['loss']))

        print("############")
        print("# EVALUATE #")
        print("############")
    
        results = self.model.evaluate(self.test_gen, return_dict=True, verbose=1)
        wandb.log(results)


if __name__ == "__main__":
    print("CONFIGURATION:")
    cf = get_config()
    
    for k, v in cf.items():
        print(k)
        print(v)
    
    print()
    
    print("DATASET SPLIT:")
    trn, vld, tst = split_dataset(cf['VOLUME']['input'])
    print("TRAINING")
    print(trn)
    print("VALIDATION")
    print(vld)
    print("TESTING")
    print(tst)