# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
import numpy as np

# local imports
import utils
import carreno.nn.metrics as mtc
from carreno.nn.unet import UNet, encoder_trainable
from carreno.nn.generators import Generator
import carreno.processing.transforms as tfs

# adjustable settings
plt.rc('font', size=18)
config         = utils.get_config()
out_model_dir  = config['DIR']['model']
out_model_name = "unet2d_base.hdf5"
wdb_project    = "unet2d_base"

params = {
    'ndim'     : 2,                  # 2 or 3
    'shape'    : [1, 96, 96],        # data shape
    'depth'    : 4,                  # unet depth
    'nfeat'    : 64,                 # nb feature for first conv layer
    'lr'       : [0.001, 0, 0.001],  # learning rate [init, min, max]
    'warmup'   : 1,                  # nb epoch for lr warmup
    'decay'    : 50,                 # nb epoch for lr decay
    'bsize'    : 32,                 # batch size
    'nepoch'   : 50,                 # number of epoch
    'scaler'   : 'norm',             # "norm" or "stand"
    'label'    : 'hard',             # "hard" or "soft" input
    'sample'   : False,              # use sample weights for unbalanced data
    'weight'   : None,               # use class weights for unbalanced data (list of classes weight or None)
    'order'    : 'before',           # where to put batch norm
    'ncolor'   : 1,                  # color depth for input
    'act'      : 'relu',             # activation
    'loss'     : 'dice',             # loss function
    'topact'   : 'softmax',          # top activation
    'dropout'  : 0.0,                # dropout rate
    'backbone' : 'unet',             # "unet" or "vgg16"
    'pretrn'   : False,              # pretrained encoder on imagenet
    'dupe'     : 48                  # nb d'usage pour un volume dans une époque
}


def setup_model(verbose=0):
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

    # freeze encoder
    if params['pretrn']:
        encoder_trainable(model, False)

    if verbose:
        model.summary()
        
    return model


def setup_files(verbose=0):
    # get file paths
    trn, vld, tst = utils.split_dataset(config['VOLUME']['rl_input'])
    fullpath = lambda dir, files : [os.path.join(dir, name) for name in files] * params['dupe']
    hard_label = params['label'] == 'hard'
    x_train = fullpath(config['VOLUME']['rl_input'], trn)
    y_train = fullpath(config['VOLUME']['target' if hard_label else 'soft_target'], trn)
    x_valid = fullpath(config['VOLUME']['rl_input'],  vld)
    y_valid = fullpath(config['VOLUME']['target' if hard_label else 'soft_target'], vld)
    x_test  = fullpath(config['VOLUME']['rl_input'],  tst)
    y_test  = fullpath(config['VOLUME']['target'], tst)
    w_train = fullpath(config['VOLUME']['weight' if hard_label else 'soft_weight'], trn) if params['sample'] else None
    w_valid = fullpath(config['VOLUME']['weight' if hard_label else 'soft_weight'], vld) if params['sample'] else None

    if verbose:
        print("Training dataset")
        if params['sample']:
            print("-nb of instances :", len(x_train), "/", len(y_train), "/", len(w_train))
            print("Validation dataset")
            print("-nb of instances :", len(x_valid), "/", len(y_valid), "/", len(w_valid))
        else:
            print("-nb of instances :", len(x_train), "/", len(y_train))
            print("Validation dataset")
            print("-nb of instances :", len(x_valid), "/", len(y_valid))
        print("Testing dataset")
        print("-nb of instances :", len(x_test), "/",  len(y_test))
    return x_train, y_train, w_train, x_valid, y_valid, w_valid, x_test, y_test


def setup_generators(x_train, y_train, w_train,
                     x_valid, y_valid, w_valid,
                     x_test, y_test, verbose=0):
    train_aug, test_aug = utils.augmentations(shape=params['shape'],
                                              norm_or_std=True if params['scaler'] == 'norm' else False,
                                              is_2D=params['ndim'] == 2,
                                              n_color_ch=params['ncolor'])

    train_gen = Generator(x_train,
                          y_train,
                          weight=w_train if params['sample'] else None,
                          size=params['bsize'],
                          augmentation=train_aug,
                          shuffle=True)
    valid_gen = Generator(x_valid,
                          y_valid,
                          weight=w_valid if params['sample'] else None,
                          size=params['bsize'],
                          augmentation=test_aug,
                          shuffle=False)  # make sure all patches fit in epoch
    test_gen  = Generator(x_test,
                          y_test,
                          weight=None,
                          size=params['bsize'],
                          augmentation=test_aug,
                          shuffle=False)
    
    if verbose:
        print("Generator info")
        print("-length :", len(train_gen))
        
        batch0 = train_gen[0]
        print("-batch output shape :", batch0[0].shape, "/", batch0[1].shape, end="")
        if len(batch0) > 2:
            print(" /", batch0[2].shape)
        else:
            print()

        print("-batch visualization :")
        
        # cool visuals which are a joy to debug
        nb_columns = len(batch0)
        btc_size   = params['bsize']
        nb_lines   = min(btc_size, 2)
        for i in range(nb_lines):
            for j, k in zip(range(1, nb_columns+1), ['x', 'y', 'w']):
                one_input = batch0[j-1].numpy()
                plt.subplot(nb_lines, nb_columns, i*nb_columns+j)
                plt.title(k + " " + str(i))
                if params['ndim'] == 2:
                    plt.imshow(one_input[i], vmin=one_input.min(), vmax=one_input.max())
                else:
                    hslc = one_input[i].shape[0] // 2
                    plt.imshow(one_input[i][hslc], vmin=one_input.min(), vmax=one_input.max())
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    return train_gen, valid_gen, test_gen


def plot_metrics(path, history, verbose=0):
    # metrics display (acc, loss, etc.)
    graph_path = path

    def get_color():
        while 1:
            for j in ['b', 'y', 'r', 'g']:
                yield j
    color = get_color()

    epochs = np.array(history.history['epoch']) + 1
    loss_hist = history.history['loss']
    val_loss_hist = history.history['val_loss']
    plt.plot(epochs, loss_hist, next(color), label='trn')
    plt.plot(epochs, val_loss_hist, next(color), label='val')
        
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_path + "_loss.svg", format="svg")
    plt.savefig(graph_path + "_loss.png")
    plt.show(block=False) if verbose else plt.clf()
    plt.pause(3)
    plt.close()
    
    epochs = np.array(history.history['epoch']) + 1
    dice_hist = history.history['dice']
    val_dice_hist = history.history['val_dice']
    plt.plot(epochs, dice_hist, next(color), label='trn')
    plt.plot(epochs, val_dice_hist, next(color), label='val')
        
    plt.title('Training and validation Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_path + "_dice.svg", format="svg")
    plt.savefig(graph_path + "_dice.png")
    plt.show(block=False) if verbose else plt.clf()
    plt.pause(3)
    plt.close()

    epochs = np.array(history.history['epoch']) + 1
    dice_hist = history.history['cldice']
    val_dice_hist = history.history['val_cldice']
    plt.plot(epochs, dice_hist, next(color), label='trn')
    plt.plot(epochs, val_dice_hist, next(color), label='val')

    plt.title('Training and validation DiceClDice')
    plt.xlabel('Epochs')
    plt.ylabel('DiceClDice Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_path + "_dicecldice.svg", format="svg")
    plt.savefig(graph_path + "_dicecldice.png")
    plt.show(block=False) if verbose else plt.clf()
    plt.pause(3)
    plt.close()


def main(verbose=0):
    print("###############")
    print("# SETUP MODEL #")
    print("###############")
    model = setup_model(verbose)
    
    # setup wandb
    wandb.init(project=wdb_project, config=params)

    print("###############")
    print("# SETUP FILES #")
    print("###############")
    x_train, y_train, w_train, x_valid, y_valid, w_valid, x_test, y_test = setup_files(verbose)

    print("#################")
    print("# SETUP LOADERS #")
    print("#################")
    train_gen, valid_gen, test_gen = setup_generators(x_train, y_train, w_train,
                                                      x_valid, y_valid, w_valid,
                                                      x_test, y_test, verbose)

    print("#############")
    print("# CALLBACKS #")
    print("#############")

    out_model_path = os.path.join(out_model_dir, out_model_name)
    Path(out_model_dir).mkdir(parents=True, exist_ok=True)

    # callbacks
    monitor, mode = 'val_loss', 'min'
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
                                                    options=None,
                                                    verbose=1)
    
    nsteps_per_epoch = len(train_gen)
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=params['lr'][0],
        decay_steps=params['decay'] * nsteps_per_epoch,
        alpha=params['lr'][1],
        warmup_target=params['lr'][2],
        warmup_steps=params['warmup'] * nsteps_per_epoch)
    optim = tf.keras.optimizers.legacy.Adam(learning_rate=schedule)
    current_epoch = 0

    iters = 10
    ndim  = params['ndim']
    pad   = 2
    smooth = 1e-6
    dice       = mtc.Dice(class_weight=params['weight'])
    cldice     = mtc.ClDice(    iters=iters, ndim=ndim, mode=pad, smooth=smooth, class_weight=params['weight'])
    dicecldice = mtc.DiceClDice(iters=iters, ndim=ndim, mode=pad, smooth=smooth, class_weight=params['weight'])
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
                                             smooth=smooth,
                                             class_weight=params['weight']).loss
    else:
        raise NotImplementedError

    def compile_n_fit(train_gen, valid_gen, start_epoch, log=True):
        model.compile(optimizer=optim,
                      loss=loss_fn,
                      weighted_metrics=[dice.coefficient, cldice.coefficient, dicecldice.coefficient],
                      sample_weight_mode="temporal", run_eagerly=True)
  
        callbacks = [early_stop] + ([checkpoint, metrics_logger] if log else [])
            
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
    history, current_epoch = compile_n_fit(train_gen,
                                           valid_gen,
                                           current_epoch)
    plot_metrics(graph_path, history, verbose)

    print("############")
    print("# EVALUATE #")
    print("############")
    results = model.evaluate(test_gen, return_dict=True, verbose=1)
    wandb.log(results)


if __name__ == "__main__":
    main(verbose=1)
