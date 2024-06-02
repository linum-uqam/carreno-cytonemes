# -*- coding: utf-8 -*-
import yaml
import os
import tifffile as tif
import numpy as np
import random
import wandb
import tensorflow as tf
from skimage.restoration import estimate_sigma
from sklearn.model_selection import train_test_split
from pathlib import Path
import carreno.processing.transforms as tfs
import carreno.nn.metrics as mtc
from carreno.nn.unet import UNet, encoder_trainable
from carreno.nn.generators import Generator
from carreno.pipeline.pipeline import Threshold, UNet2D, UNet3D


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


class UNetSweeper():
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
        self.wandb_artifact = wandb_artifact
        self.params = {
            'ndim':     ndim,                                  # 2 or 3
            'shape':    [1 if ndim == 2 else 48, 96, 96],      # data shape
            'depth':    4,                                     # unet depth
            'nfeat':    64,                                    # nb feature for first conv layer
            'nepoch':   self.prj_config['TRAINING']['epoch'],  # nb epoch for training
            'lr':       [0.001, 0, 0.001],                     # learning rate [init, min, max]
            'warmup':   0,                                     # nb epoch for lr warmup
            'bsize':    32 if ndim == 2 else 1,                # batch size
            'scaler':   'norm',                                # normalize or standardize
            'label':    'hard',                                # hard or soft input
            'weight':   False,                                 # use sample weights for unbalanced data
            'classw':   None,                                  # use class weights for unbalanced data
            'order':    'before',                              # where to put batch norm
            'ncolor':   1,                                     # color depth for input
            'act':      'relu',                                # activation
            'loss':     'dice',                                # loss function
            'topact':   'softmax',                             # top activation
            'dropout':  0.0,                                   # dropout rate
            'backbone': 'base',                                # base or unet
            'pretrn':   False,                                 # pretrained on imagenet
            'slftrn':   False,                                 # self train on unlabeled data
            'dupe':     48 if ndim == 2 else 1                 # number of data duplication to fit batch size
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
        x_train = fullpath(self.prj_config['VOLUME']['rl_input'],  trn)
        y_train = fullpath(self.prj_config['VOLUME']['target' if hard_label else 'soft_target'], trn)
        w_train = fullpath(self.prj_config['VOLUME']['weight' if hard_label else 'soft_weight'], trn) if self.params['weight'] else None
        x_valid = fullpath(self.prj_config['VOLUME']['rl_input'],  vld)
        y_valid = fullpath(self.prj_config['VOLUME']['target'], vld)
        x_test  = fullpath(self.prj_config['VOLUME']['rl_input'],  tst)
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
        model_name = self.swp_config['project'] + param_str + ".keras"
        model_path = os.path.join(self.prj_config['DIR']['model'], model_name)
        Path(self.prj_config['DIR']['model']).mkdir(parents=True, exist_ok=True)

        # callbacks
        monitor, mode = self.swp_config['metric']['name'], self.swp_config['metric']['goal'][:3]
        early_stop = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                      mode=mode,
                                                      patience=10,
                                                      restore_best_weights=True,
                                                      verbose=1)
        metrics_logger = wandb.keras.WandbMetricsLogger()
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

        total_steps = len(self.train_gen) * self.params['nepoch']
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.params['lr'][0],
            decay_steps=total_steps - self.params['warmup'],
            alpha=self.params['lr'][1],
            warmup_target=self.params['lr'][2],
            warmup_steps=self.params['warmup'])
        optim = tf.keras.optimizers.legacy.Adam(learning_rate=schedule)

        iters = 10
        ndim  = ndim=self.params['ndim']
        pad   = 2
        cls   = slice(1,2)  # only skeletonize cyto (tried, but gives awful result for cldice)
        dice       = mtc.Dice(class_weight=self.params['classw'])
        cldice     = mtc.ClDice(    iters=iters, ndim=ndim, mode=pad, class_weight=self.params['classw'])
        dicecldice = mtc.DiceClDice(iters=iters, ndim=ndim, mode=pad, class_weight=self.params['classw'])
        if self.params['loss'] == "dice":
            loss_fn = dice.loss
        elif self.params['loss'] == "dicecldice":
            loss_fn = dicecldice.loss
        elif self.params['loss'] == "cedice":
            loss_fn = mtc.CeDice(class_weight=self.params['classw']).loss
        elif self.params['loss'] == "adawing":
            loss_fn = mtc.AdaptiveWingLoss(class_weight=self.params['classw']).loss
        elif self.params['loss'] == "cldiceadawing":
            loss_fn = mtc.ClDiceAdaptiveWingLoss(iters=iters, ndim=ndim, mode=pad, class_weight=self.params['classw']).loss
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
                                  epochs=self.params['nepoch'],
                                  initial_epoch=initial_epoch,
                                  callbacks=callbacks,
                                  verbose=1)

        if self.params['slftrn']:
            print("##############")
            print("# SELF-TRAIN #")
            print("##############")
            other_optim = tf.keras.optimizers.legacy.Adam(learning_rate=schedule)
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


class RestoreSweeper():
    def __init__(self, config):
        """
        Setup sweeps for wandb.
        Parameters
        ----------
        config : dict
            Sweeps configuration.
            Refer to Wandb for format https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
        """
        self.prj_config = get_config()
        self.swp_config = config
        self.params = {
            'psf':       False,
            'iteration': 50,
            'nlm_size':  0,
            'nlm_dist':  0,
            'r':         50,
            'amount':    10,
            'btw_h':     1,
            'btw_l':     1,
            'btw_s':     0,
            'md_ax0':    1,
            'md_ax12':   3
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
        print("# SET LOADERS #")
        print("###############")

        inputs  = list(os.listdir(self.prj_config['VOLUME']['input']))
        fullpath = lambda dir, files : [os.path.join(dir, name) for name in files]
        xs = fullpath(self.prj_config['VOLUME']['input'],  inputs)
        outputs = fullpath(self.prj_config['VOLUME']['restore'], inputs)
        psf = tif.imread(os.path.join(self.prj_config['DIR']['psf'], "Averaged PSF.tif"))
        
        print("#############")
        print("# RESTORING #")
        print("#############")
        pipeline = Threshold()
        
        for x, out in zip(xs, outputs):
            y = pipeline.restore(tif.imread(x),
                                 psf=psf if self.params['psf'] else None,
                                 iteration=self.params['iteration'],
                                 nlm_size=self.params['nlm_size'],
                                 nlm_dist=self.params['nlm_dist'],
                                 r=self.params['r'],
                                 amount=self.params['amount'],
                                 md_size=[self.params['md_ax0'], self.params['md_ax12'], self.params['md_ax12']],
                                 butterworth=[self.params['btw_h'], self.params['btw_l'], self.params['btw_s']])
            tif.imwrite(out, y, photometric='minisblack')

        print("############")
        print("# EVALUATE #")
        print("############")
        noise = np.zeros((len(outputs)))
        
        for i in range(noise.shape[0]):
            noise[i] = estimate_sigma(tif.imread(outputs[i]))

        #log_dict = self.params.copy()
        #log_dict.update({"noise" : noise.mean()})
        wandb.log({"noise" : noise.mean()})


class ThresholdSweeper():
    def __init__(self, config):
        """
        Setup sweeps for wandb.
        Parameters
        ----------
        config : dict
            Sweeps configuration.
            Refer to Wandb for format https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
        """
        self.prj_config = get_config()
        self.swp_config = config
        self.params = {
            'psf':       False,
            'iteration': 50,
            'nlm_size':  0,
            'nlm_dist':  0,
            'r':         50,
            'amount':    10,
            'btw_h':     1,
            'btw_l':     1,
            'btw_s':     0,
            'md_ax0':    1,
            'md_ax12':   3,
            'ro':        1,
            'rc':        1
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
        print("# SET LOADERS #")
        print("###############")

        inputs  = list(os.listdir(self.prj_config['VOLUME']['input']))
        fullpath = lambda dir, files : [os.path.join(dir, name) for name in files]
        xs = fullpath(self.prj_config['VOLUME']['input'], inputs)
        ys = fullpath(self.prj_config['VOLUME']['target'], inputs)
        outputs = fullpath(self.prj_config['VOLUME']['threshold'], inputs)
        psf = tif.imread(os.path.join(self.prj_config['DIR']['psf'], "Averaged PSF.tif"))
        
        print("###########")
        print("# SEGMENT #")
        print("###########")
        pipeline = Threshold()
        dice_fn = mtc.Dice().coefficient
        dicecldice_fn = mtc.DiceClDice(iters=10, ndim=3, mode=2).coefficient
        dice_scores = np.zeros((len(outputs)))
        dicecldice_scores = np.zeros((len(outputs)))

        for i in range(len(outputs)):
            x = tif.imread(xs[i])
            y = tif.imread(ys[i])
            out = outputs[i]
            d = pipeline.restore(x,
                                 psf=psf if self.params['psf'] else None,
                                 iteration=self.params['iteration'],
                                 nlm_size=self.params['nlm_size'],
                                 nlm_dist=self.params['nlm_dist'],
                                 r=self.params['r'],
                                 amount=self.params['amount'],
                                 md_size=[self.params['md_ax0'], self.params['md_ax12'], self.params['md_ax12']],
                                 butterworth=[self.params['btw_h'], self.params['btw_l'], self.params['btw_s']])
            p = pipeline.segmentation(d,
                                      ro=self.params['ro'],
                                      rc=self.params['rc'])
            
            ty = tf.convert_to_tensor(np.expand_dims(y, 0), tf.float32)
            tp = tf.convert_to_tensor(np.expand_dims(p, 0), tf.float32)
            dice_scores[i] = dice_fn(ty, tp)
            dicecldice_scores[i] = dicecldice_fn(ty, tp)
            
            tif.imwrite(out, p, photometric='rgb')

        print("############")
        print("# EVALUATE #")
        print("############")
        
        #log_dict = self.params.copy()
        #log_dict.update({"noise" : noise.mean()})
        wandb.log({"dice" : dice_scores.mean(), "dicecldice" : dicecldice_scores.mean()})


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

    test_dict = {
        'method': 'grid',
        'name':   'sweep',
        'project': 'test',
        'metric': {
            'goal': 'maximize',
            'name': 'noise'
        },
        'parameters': {
        }
    }
    sweeper = ThresholdSweeper(test_dict)
    sweeper.sweep()
    