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
import carreno.processing.transforms as tfs

# adjustable settings
plt.rc('font', size=18)
config         = utils.get_config()
out_model_name = "unet2d_base.h5"
out_model_dir  = config['DIR']['model']
wdb_project    = 'unet2d_base'

params = {
    'ndim'     : 2,              # 2 or 3
    'shape'    : [1, 96, 96],    # data shape
    'depth'    : 4,              # unet depth
    'nfeat'    : 64,             # nb feature for first conv layer
    'lr'       : 0.001,          # learning rate
    'bsize'    : 32,             # batch size
    'nepoch'   : 100,            # number of epoch
    'scaler'   : 'norm',         # "norm" or "stand"
    'label'    : 'hard',         # hard or soft input
    'weight'   : True,           # use class weights for unbalanced data
    'order'    : 'before',       # where to put batch norm
    'ncolor'   : 1,              # color depth for input
    'act'      : 'relu',         # activation
    'loss'     : 'dice',         # loss function
    'topact'   : 'softmax',      # top activation
    'dropout'  : 0.0,            # dropout rate
    'backbone' : 'unet',         # "unet" or "vgg16"
    'pretrn'   : False,          # pretrained encoder on imagenet
    'slftrn'   : False,          # pretrained encoder on unlabeled
    'dupe'     : 48              # nb d'usage pour un volume dans une époque
}

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

    model.summary()

    # setup wandb
    wandb.init(project=wdb_project, config=params)

    print("###############")
    print("# SET LOADERS #")
    print("###############")

    trn, vld, tst = utils.split_dataset(config['VOLUME']['input'])
    fullpath = lambda dir, files : [os.path.join(dir, name) for name in files] * params['dupe']
    hard_label = params['label'] == 'hard'
    x_train = fullpath(config['VOLUME']['input'], trn)
    y_train = fullpath(config['VOLUME']['target' if hard_label else 'soft_target'], trn)
    w_train = fullpath(config['VOLUME']['weight' if hard_label else 'soft_weight'], trn) if params['weight'] else None
    x_valid = fullpath(config['VOLUME']['input'],  vld)
    y_valid = fullpath(config['VOLUME']['target' if hard_label else 'soft_target'], vld)
    w_valid = fullpath(config['VOLUME']['weight' if hard_label else 'soft_weight'], vld) if params['weight'] else None
    x_test  = fullpath(config['VOLUME']['input'],  tst)
    y_test  = fullpath(config['VOLUME']['target'], tst)

    unlabeled_vol = list(os.listdir(config['VOLUME']['unlabeled']))
    fullpath = lambda dir, files : [os.path.join(dir, name) for name in files] * max(1, params['dupe'] // 4)
    x_unlabeled_train = fullpath(config['VOLUME']['unlabeled'], unlabeled_vol)
    y_unlabeled_train = fullpath(config['VOLUME']['unlabeled_target' if hard_label else 'unlabeled_soft_target'], unlabeled_vol)
    w_unlabeled_train = fullpath(config['VOLUME']['unlabeled_weight' if hard_label else 'unlabeled_soft_weight'], unlabeled_vol) if params['weight'] else None
    
    print("Training dataset")
    if params['weight']:
        print("-nb of instances :", len(x_train), "/", len(y_train), "/", len(w_train))
    else:
        print("-nb of instances :", len(x_train), "/", len(y_train))

    print("Validation dataset")
    if params['weight']:
        print("-nb of instances :", len(x_valid), "/", len(y_valid), "/", len(w_valid))
    else:
        print("-nb of instances :", len(x_valid), "/", len(y_valid))

    print("Testing dataset")
    print("-nb of instances :", len(x_test), "/",  len(y_test))

    if params['slftrn']:
        print("Training unlabeled dataset")
        if params['weight']:
            print("-nb of instances :", len(x_unlabeled_train), "/", len(y_unlabeled_train), "/", len(w_unlabeled_train))
        else:
            print("-nb of instances :", len(x_unlabeled_train), "/", len(y_unlabeled_train))

    # setup data augmentation
    train_aug, test_aug = utils.augmentations(shape=params['shape'],
                                              norm_or_std=True if params['scaler'] == 'norm' else False,
                                              is_2D=params['ndim'] == 2,
                                              n_color_ch=params['ncolor'])

    # ready up the data generators
    train_gen = Generator(x_train,
                          y_train,
                          weight=w_train if params['weight'] else None,
                          size=params['bsize'],
                          augmentation=train_aug,
                          shuffle=True)
    valid_gen = Generator(x_valid,
                          y_valid,
                          weight=w_valid if params['weight'] else None,
                          size=params['bsize'],
                          augmentation=test_aug,
                          shuffle=False)  # make sure all patches fit in epoch
    test_gen  = Generator(x_test,
                          y_test,
                          weight=None,
                          size=params['bsize'],
                          augmentation=test_aug,
                          shuffle=False)
    train_unlabeled_gen = Generator(x_unlabeled_train,
                                    y_unlabeled_train,
                                    weight=w_unlabeled_train if params['weight'] else None,
                                    size=params['bsize'],
                                    augmentation=train_aug,
                                    shuffle=True)
    
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
        plt.show()

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
    
    total_steps = len(train_gen) * params['nepoch']
    schedule = tf.keras.optimizers.schedules.CosineDecay(params['lr'], decay_steps=total_steps)
    optim = tf.keras.optimizers.Adam(learning_rate=schedule)
    current_epoch = 0

    iters = 10
    ndim  = params['ndim']
    pad   = 2
    dice       = mtc.Dice()
    cldice     = mtc.ClDice(    iters=iters, ndim=ndim, mode=pad)
    dicecldice = mtc.DiceClDice(iters=iters, ndim=ndim, mode=pad)
    if params['loss'] == "dice":
        loss_fn = dice.loss
    elif params['loss'] == "dicecldice":
        loss_fn = dicecldice.loss
    elif params['loss'] == "cedice":
        loss_fn = mtc.CeDice().loss
    elif params['loss'] == "adawing":
        loss_fn = mtc.AdaptiveWingLoss().loss
    elif params['loss'] == "cldiceadawing":
        loss_fn = mtc.ClDiceAdaptiveWingLoss(iters=iters, ndim=ndim, mode=pad).loss
    else:
        raise NotImplementedError

    def compile_n_fit(train_gen, valid_gen, start_epoch, log=True):
        model.compile(optimizer=optim,
                      loss=loss_fn,
                      metrics=[dice.coefficient, cldice.coefficient, dicecldice.coefficient],
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

    histories = []

    if params['slftrn']:
        print("##############")
        print("# SELF-TRAIN #")
        print("##############")
        
        encoder_trainable(model, False)
        history_dec, current_epoch = compile_n_fit(train_unlabeled_gen,
                                                   valid_gen,
                                                   current_epoch)
        histories.append(history_dec)

    print("############")
    print("# TRAINING #")
    print("############")

    encoder_trainable(model, True)
    history, current_epoch = compile_n_fit(train_gen,
                                           valid_gen,
                                           current_epoch)
    histories.append(history)

    utils.plot_metrics(graph_path, histories, verbose)

    print("############")
    print("# EVALUATE #")
    print("############")
    results = model.evaluate(test_gen, return_dict=True, verbose=1)
    wandb.log(results)

    # pickle obj in case we can't load it after
    pickle_path = out_model_path.rsplit(".", 1)[0] + ".pkl"
    with open(pickle_path, 'wb') as f:  # open a text file
        pickle.dump(obj=model, file=f)


if __name__ == "__main__":
    main(verbose=1)