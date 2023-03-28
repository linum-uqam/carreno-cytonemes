# -*- coding: utf-8 -*-
import tensorflow as tf
import wandb
import os
from pathlib import Path
import albumentations as A
import volumentations as V

# local imports
import utils
from carreno.nn.unet import UNet
import carreno.nn.metrics as mtc
from carreno.nn.unet import encoder_trainable, switch_top
from carreno.nn.generators import get_volumes_slices, volume_generator, volume_slice_generator

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

    # must add color channel to grayscale
    is_2D = input_ndim == 4
    input_shape = config['PREPROCESS']['patch'][5 - input_ndim:] + [n_color_ch]
    
    model = UNet(shape=input_shape,
                 n_class=config['PREPROCESS']['n_cls'],
                 depth=depth,
                 n_feat=n_features,
                 dropout=dropout,
                 batch_norm=batch_order,
                 activation=activation,
                 top_activation=top_activation,
                 backbone=backbone,
                 pretrained=pretrained)
    
    encoder_trainable(model, False)

    model.summary()

    print("###############")
    print("# DATA LOADER #")
    print("###############")

    trn, vld, tst = utils.split_patches(config['PATCH']['input'])
    fullpath = lambda dir, files : [os.path.join(dir, name) for name in files]
    x_train = fullpath(config['PATCH']['input'],  trn)
    y_train = fullpath(config['PATCH']['target'], trn)
    w_train = fullpath(config['PATCH']['weight'], trn)
    x_valid = fullpath(config['PATCH']['input'],  vld)
    y_valid = fullpath(config['PATCH']['target'], vld)
    x_test  = fullpath(config['PATCH']['input'],  tst)
    y_test  = fullpath(config['PATCH']['target'], tst)

    if is_2D:
        # slice up volumes for img
        x_train = get_volumes_slices(x_train)
        y_train = get_volumes_slices(y_train)
        w_train = get_volumes_slices(w_train)
        x_valid = get_volumes_slices(x_valid)
        y_valid = get_volumes_slices(y_valid)
        x_test  = get_volumes_slices(x_test)
        y_test  = get_volumes_slices(y_test)

    print("Training dataset")
    if is_2D:
        print("-nb of volumes :",
              [j for i, j in x_train].count(0), "/",
              [j for i, j in y_train].count(0), "/",
              [j for i, j in w_train].count(0))
    print("-nb of instances :", len(x_train), "/", len(y_train), "/", len(w_train))
    
    print("Validation dataset")
    if is_2D:
        print("-nb of volumes :",
              [j for i, j in x_valid].count(0), "/",
              [j for i, j in y_valid].count(0))
    print("-nb of instances :", len(x_valid), "/", len(y_valid))
    
    print("Testing dataset")
    if is_2D:
        print("-nb of volumes :",
              [j for i, j in x_test].count(0), "/",
              [j for i, j in y_test].count(0))
    print("-nb of instances :", len(x_test), "/",  len(y_test))

    # setup data augmentation
    aug = None
    if is_2D:
        aug = A.Compose([
            A.Rotate(limit=45, interpolation=1, border_mode=4, p=0.25),
            A.RandomRotate90((1, 2), p=0.25),
            A.Flip(0, p=0.25),
            A.Flip(1, p=0.25)
        ], additional_targets={"weight":"mask"}, p=1)
    else:
        aug = V.Compose([
            V.Rotate((-45, 45), (0,0), (0, 0), border_mode='reflect', p=0.25),
            V.RandomRotate90((1, 2), p=0.25),
            V.Flip(0, p=0.25),
            V.Flip(1, p=0.25)
        ], targets=[['image','mask','weight']], p=1)

    # ready up the data generators
    batch_size = config['TRAINING']['batch2D' if is_2D else 'batch3D']
    generator_fn = volume_slice_generator if is_2D else volume_generator
    train_gen = generator_fn(x_train,
                             y_train,
                             weight=w_train,
                             size=batch_size,
                             augmentation=aug,
                             nb_color_ch=n_color_ch,
                             shuffle=True)
    valid_gen = generator_fn(x_valid,
                             y_valid,
                             weight=None,  # not used for validation since it would cause a bias on best saved model
                             size=batch_size,
                             augmentation=None,
                             nb_color_ch=n_color_ch,
                             shuffle=True)  # I'm kind of torn on putting shuffle on validation data, but otherwise, the extra patches are never used  
    test_gen  = generator_fn(x_test,
                             y_test,
                             weight=None,
                             size=batch_size,
                             augmentation=None,
                             nb_color_ch=n_color_ch,
                             shuffle=False)

    print("############")
    print("# TRAINING #")
    print("############")

    LR         = 0.001
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

    model.compile(optimizer=optim,
                  loss=mtc.dice_loss,
                  metrics=mtc.dice_score(smooth=1.),
                  sample_weight_mode="temporal")

    model.fit(train_gen,
              validation_data=valid_gen,
              steps_per_epoch=len(train_gen),
              validation_steps=len(valid_gen),
              batch_size=config['TRAINING']['batch2D'],
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