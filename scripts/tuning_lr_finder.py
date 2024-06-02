# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import wandb
from pathlib import Path

# local imports
import utils
import carreno.nn.metrics as mtc
from carreno.nn.unet import UNet, encoder_trainable
from carreno.nn.generators import Generator
from carreno.nn.one_cycle_callback import LRFinder, OneCycleLR
import carreno.processing.transforms as tfs

# adjustable settings
plt.rc('font', size=18)
config         = utils.get_config()
out_model_dir  = config['DIR']['model']
out_model_name = "unet2d_lr_finder.h5"
wdb_project    = 'unet2d_lr_finder'

params = {
    'ndim'     : 2,              # 2 or 3
    'shape'    : [1, 192, 192],  # data shape
    'depth'    : 4,              # unet depth
    'nfeat'    : 64,             # nb feature for first conv layer
    'lr'       : 0.001,          # learning rate
    'bsize'    : 16,             # batch size
    'nepoch'   : 1,              # number of epoch
    'scaler'   : 'stand',        # "norm" or "stand"
    'label'    : 'soft',         # "hard" or "soft" input
    'weight'   : [0.37,38.54,4], # use class weights for unbalanced data
    'order'    : 'after',        # where to put batch norm
    'ncolor'   : 3,              # color depth for input
    'act'      : 'gelu',         # activation
    'loss'     : 'dicecldice',   # loss function [...]
    'topact'   : 'softmax',      # top activation
    'dropout'  : 0.0,            # dropout rate
    'backbone' : 'vgg16',        # "unet" or "vgg16"
    'pretrn'   : True,           # pretrained encoder on imagenet
    'dupe'     : 64              # nb of usage for a volume per epoch
}

dir_name = "testMaxLR/"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def main(verbose=0):
    print("###############")
    print("# BUILD MODEL #")
    print("###############")

    model = UNet(shape=params['shape'][1:] + [params['ncolor']],
                 n_class=config['PREPROCESS']['n_cls'],
                 depth=params['depth'],
                 n_feat=params['nfeat'],
                 dropout=params['dropout'],
                 norm_order=params['order'],
                 activation=params['act'],
                 top_activation=params['topact'],
                 backbone=None if params['backbone'] == "unet" else params['backbone'],
                 pretrained=params['pretrn'])

    if True:
        model.summary()

    # setup wandb
    wandb.init(project=wdb_project, config=params)

    print("###############")
    print("# SET LOADERS #")
    print("###############")

    trn, vld, tst = utils.split_dataset(config['VOLUME']['input'])
    fullpath = lambda dir, files : [os.path.join(dir, name) for name in files] * params['dupe']
    hard_label = params['label'] == 'hard'
    x_train = fullpath(config['VOLUME']['rl_input'], trn)
    y_train = fullpath(config['VOLUME']['target' if hard_label else 'soft_target'], trn)
    x_valid = fullpath(config['VOLUME']['rl_input'],  vld)
    y_valid = fullpath(config['VOLUME']['target' if hard_label else 'soft_target'], vld)
    x_test  = fullpath(config['VOLUME']['rl_input'],  tst)
    y_test  = fullpath(config['VOLUME']['target'], tst)

    unlabeled_vol = list(os.listdir(config['VOLUME']['unlabeled']))
    fullpath = lambda dir, files : [os.path.join(dir, name) for name in files] * max(1, params['dupe'])
    x_unlabeled_train = fullpath(config['VOLUME']['rl_unlabeled'], unlabeled_vol)
    y_unlabeled_train = fullpath(config['VOLUME']['unlabeled_target' if hard_label else 'unlabeled_soft_target'], unlabeled_vol)
    
    print("Training dataset")
    print("-nb of instances :", len(x_train), "/", len(y_train))
    print("Validation dataset")
    print("-nb of instances :", len(x_valid), "/", len(y_valid))
    print("Testing dataset")
    print("-nb of instances :", len(x_test), "/",  len(y_test))

    # setup data augmentation
    train_aug, test_aug = utils.augmentations(shape=params['shape'],
                                              norm_or_std=True if params['scaler'] == 'norm' else False,
                                              is_2D=params['ndim'] == 2,
                                              n_color_ch=params['ncolor'])

    # ready up the data generators
    validation_gen = Generator(x_train + x_valid + x_test,
                          y_train + y_valid + y_test,
                          size=params['bsize'],
                          augmentation=train_aug,
                          shuffle=True)
    # ready up the data generators
    train_gen = Generator(x_train,
                          y_train,
                          size=params['bsize'],
                          augmentation=train_aug,
                          shuffle=True)
    valid_gen = Generator(x_valid,
                          y_valid,
                          size=params['bsize'],
                          augmentation=test_aug,
                          shuffle=False)  # make sure all patches fit in epoch

    print("#############")
    print("# CALLBACKS #")
    print("#############")

    out_model_path = os.path.join(out_model_dir, out_model_name)
    Path(out_model_dir).mkdir(parents=True, exist_ok=True)

    # callbacks
    monitor, mode = 'val_dicecldice', 'max'
    patience = 10
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                  mode=mode,
                                                  patience=patience,
                                                  restore_best_weights=True,
                                                  verbose=1)
    metrics_logger = wandb.keras.WandbMetricsLogger()
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=out_model_path,
                                                    monitor=monitor,
                                                    mode=mode,
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    verbose=1)
    
    num_samples = len(validation_gen) * params['bsize']
    minimum_lr = 0.00005
    maximum_lr = 1
    lr_callback = LRFinder(num_samples, params['bsize'],
                       minimum_lr, maximum_lr,
                       validation_data=validation_gen,
                       validation_sample_rate=5,
                       lr_scale='linear', save_dir=dir_name)
    
    total_steps = len(validation_gen) * params['nepoch']
    #schedule = tf.keras.optimizers.schedules.CosineDecay(params['lr'], decay_steps=total_steps)
    #optim = tf.keras.optimizers.Adam(learning_rate=schedule)
    optim = tf.keras.optimizers.Adam() #tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    current_epoch = 0

    iters = 10
    ndim  = params['ndim']
    pad   = 2
    dice       = mtc.Dice(class_weight=params['weight'])
    cldice     = mtc.ClDice(    iters=iters, ndim=ndim, mode=pad, class_weight=params['weight'])
    dicecldice = mtc.DiceClDice(iters=iters, ndim=ndim, mode=pad, class_weight=params['weight'])
    if params['loss'] == "dice":
        loss_fn = dice.loss
    elif params['loss'] == "dicecldice":
        loss_fn = dicecldice.loss
    elif params['loss'] == "cedice":
        loss_fn = mtc.CeDice(class_weight=params['weight']).loss
    elif params['loss'] == "adawing":
        loss_fn = mtc.AdaptiveWingLoss(class_weight=params['weight']).loss
    elif params['loss'] == "cldiceadawing":
        loss_fn = mtc.ClDiceAdaptiveWingLoss(iters=iters,
                                             ndim=ndim,
                                             mode=pad,
                                             class_weight=params['weight']).loss
    else:
        raise NotImplementedError

    def compile_n_fit(train_gen, valid_gen, start_epoch, log=True):
        model.compile(optimizer=optim,
                      loss=loss_fn,
                      metrics=[dice.coefficient, cldice.coefficient, dicecldice.coefficient],
                      sample_weight_mode="temporal", run_eagerly=True)
  
        callbacks = [checkpoint, early_stop, lr_callback] + ([metrics_logger] if log else [])
            
        hist = model.fit(train_gen,
                         validation_data=valid_gen,
                         steps_per_epoch=len(train_gen),
                         validation_steps=len(valid_gen),
                         batch_size=params['bsize'],
                         epochs=params['nepoch'],
                         initial_epoch=start_epoch,
                         callbacks=callbacks,
                         verbose=1)
        
        # save epoch history
        nb_epoch = len(hist.history['loss'])
        final_epoch = start_epoch + nb_epoch
        hist.history['epoch'] = list(range(start_epoch, final_epoch))
        
        return hist, final_epoch

    graph_path = out_model_path.rsplit(".", 1)[0]

    print("############")
    print("# TRAINING #")
    print("############")

    encoder_trainable(model, True)
    history, current_epoch = compile_n_fit(train_gen,
                                           valid_gen,
                                           current_epoch)
    lr_callback.plot_schedule()


if __name__ == "__main__":
    main(verbose=1)