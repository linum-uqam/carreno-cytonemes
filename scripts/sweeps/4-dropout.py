# -*- coding: utf-8 -*-
import tensorflow as tf
import wandb
import os
import numpy as np
from pathlib import Path

# local imports
import utils
from carreno.nn.unet import UNet
import carreno.nn.metrics as mtc
from carreno.nn.unet import encoder_trainable, switch_top
from carreno.nn.generators import Generator

sweep_config = {
    'method': 'grid',
    'name':   'sweep',
    'project': 'unet2d_dropout',
    'metric': {
        'goal': 'maximize',
        'name': 'val_dice'
    },
    'parameters': {
        'dropout': {'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
    }
}

config = utils.get_config()

def training():
    print("###############")
    print("# BUILD MODEL #")
    print("###############")
    
    wandb.init()
    
    # UNet settings
    input_ndim     = 4
    depth          = 4
    n_features     = 64
    dropout        = wandb.config.dropout
    batch_order    = 'after'
    activation     = 'relu'
    top_activation = 'softmax'
    backbone       = 'vgg16'
    n_color_ch     = 3
    pretrained     = True
    LR             = 0.001
    batch_size     = config['TRAINING']['batch2D' if input_ndim == 4 else 'batch3D']
    
    # must add color channel to grayscale
    is_2D = input_ndim == 4
    input_shape = config['PREPROCESS']['patch'][5 - input_ndim:]
    
    model = UNet(shape=input_shape + [n_color_ch],
                 n_class=config['PREPROCESS']['n_cls'],
                 depth=depth,
                 n_feat=n_features,
                 dropout=dropout,
                 batch_norm=batch_order,
                 activation=activation,
                 top_activation=top_activation,
                 backbone=backbone,
                 pretrained=pretrained)
    
    model.summary()

    print("###############")
    print("# DATA LOADER #")
    print("###############")

    trn, vld, tst = utils.split_dataset(config['VOLUME']['input'])
    fullpath = lambda dir, files : [os.path.join(dir, name) for name in files] * batch_size
    x_train = fullpath(config['VOLUME']['input'],  trn)
    y_train = fullpath(config['VOLUME']['target'], trn)
    w_train = fullpath(config['VOLUME']['weight'], trn)
    x_valid = fullpath(config['VOLUME']['input'],  vld)
    y_valid = fullpath(config['VOLUME']['target'], vld)
    x_test  = fullpath(config['VOLUME']['input'],  tst)
    y_test  = fullpath(config['VOLUME']['target'], tst)

    print("Training dataset")
    print("-nb of instances :", len(x_train), "/", len(y_train), "/", len(w_train))
    
    print("Validation dataset")
    print("-nb of instances :", len(x_valid), "/", len(y_valid))
    
    print("Testing dataset")
    print("-nb of instances :", len(x_test), "/",  len(y_test))

    # setup data augmentation
    train_aug, test_aug = utils.augmentations(shape=([1] if is_2D else []) + input_shape,
                                              norm_or_std=1,
                                              is_2D=is_2D,
                                              n_color_ch=n_color_ch)

    # ready up the data generators
    train_gen = Generator(x_train,
                          y_train,
                          weight=w_train,
                          size=batch_size,
                          augmentation=train_aug,
                          shuffle=True)
    valid_gen = Generator(x_valid,
                          y_valid,
                          weight=None,   # not used for validation since it would cause a bias on best saved model
                          size=batch_size,
                          augmentation=test_aug,
                          shuffle=True)  # I'm kind of torn on putting shuffle on validation data, but otherwise, the extra patches are never used  
    test_gen  = Generator(x_test,
                          y_test,
                          weight=None,
                          size=batch_size,
                          augmentation=test_aug,
                          shuffle=False)

    print("############")
    print("# TRAINING #")
    print("############")

    model_name = sweep_config['project'] + "-" + str(dropout) + ".h5"
    model_path = os.path.join(config['DIR']['model'], model_name)
    Path(config['DIR']['model']).mkdir(parents=True, exist_ok=True)

    # callbacks
    monitor, mode = sweep_config['metric']['name'], sweep_config['metric']['goal'][:3]
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                  mode=mode,
                                                  patience=10,
                                                  restore_best_weights=True,
                                                  verbose=1)
    wandb_checkpoint = wandb.keras.WandbModelCheckpoint(filepath=model_path,
                                                        monitor='val_dice',
                                                        mode='max',
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        verbose=1)
    metrics_logger   = wandb.keras.WandbMetricsLogger()

    total_steps = len(train_gen) * config['TRAINING']['epoch']
    schedule = tf.keras.optimizers.schedules.CosineDecay(LR, decay_steps=total_steps)
    optim = tf.keras.optimizers.Adam(learning_rate=schedule)

    # Train decoder
    encoder_trainable(model, False)
    model.compile(optimizer=optim,
                  loss=mtc.dice_loss,
                  metrics=mtc.dice_score(smooth=1.),
                  sample_weight_mode="temporal")

    model.fit(train_gen,
              validation_data=valid_gen,
              steps_per_epoch=len(train_gen),
              validation_steps=len(valid_gen),
              batch_size=batch_size,
              epochs=config['TRAINING']['epoch'],
              verbose=1,
              callbacks=[
                  early_stop,
                  wandb_checkpoint,
                  metrics_logger
              ])

    # Train encoder
    encoder_trainable(model, True)
    model.compile(optimizer=optim,
                  loss=mtc.dice_loss,
                  metrics=mtc.dice_score(smooth=1.),
                  sample_weight_mode="temporal")

    model.fit(train_gen,
              validation_data=valid_gen,
              steps_per_epoch=len(train_gen),
              validation_steps=len(valid_gen),
              batch_size=batch_size,
              epochs=config['TRAINING']['epoch'],
              verbose=1,
              callbacks=[
                  early_stop,
                  wandb_checkpoint,
                  metrics_logger
              ])

    print("############")
    print("# EVALUATE #")
    print("############")

    results = model.evaluate(test_gen, return_dict=True, verbose=1)
    wandb.log(results)


def main():
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=training)  


if __name__ == "__main__":
    main()