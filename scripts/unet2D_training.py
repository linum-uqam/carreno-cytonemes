# -*- coding: utf-8 -*-
import numpy as np
import tifffile as tif
import os
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from skimage.transform import resize

# local imports
import utils
import carreno.nn.callbacks as cb
from carreno.nn.unet import UNet, encoder_trainable
from carreno.nn.generators import volume_slice_generator
from carreno.nn.metrics import dice_score, bce_dice_loss
from carreno.processing.patches import volume_pred_from_img

config = utils.get_config()

# main settings
backbone       = config['TRAINING']['backbone'][0]
depth          = 4
n_features     = 64
soft_labels    = 1  # use soft labels over hard

# visualization
test_split        = 0  # show if data split info
test_generator    = 0  # show training gen output
test_architecture = 0  # show model summary
test_prediction   = 1  # show a few prediction slices

y, w        = ('sy', 'sw') if soft_labels else ('y', 'w')
input_shape = config['PREPROCESS']['patch'][1:] + [1]
test_volume = os.path.join(config['VOLUME']['input'],    "slik6.tif")
test_target = os.path.join(config['VOLUME']['target'],   Path(test_volume).name)
model_path  = os.path.join(config['TRAINING']['output'], "unet2D_test.h5")
info_path   = os.path.join(config['TRAINING']['output'], Path(model_path).name.split('.')[0])

# parameters and hyperparameters
LR = 0.001

# metrics
metrics = [dice_score(smooth=1.)]

# callbacks
early_stop = cb.early_stop(metric='val_dice',
                           mode='max',
                           patience=config['TRAINING']['patience'])
model_checkpoint = cb.model_checkpoint(model_path,
                                       metric='val_dice',
                                       mode='max')


def main():
    # split data between training, validation and test
    train, valid, test = utils.split_dataset(0.2, 0.2)
    
    # slice up volumes for img
    x_train = utils.get_volumes_slices(train['x'])
    y_train = utils.get_volumes_slices(train[y])
    w_train = utils.get_volumes_slices(train[w])
    x_valid = utils.get_volumes_slices(valid['x'])
    y_valid = utils.get_volumes_slices(valid[y])
    w_valid = utils.get_volumes_slices(valid[w])
    x_test  = utils.get_volumes_slices(test['x'])
    y_test  = utils.get_volumes_slices(test[y])
    w_test  = utils.get_volumes_slices(test[w])
    
    if test_split:
        print("Training dataset")
        print("-nb of volumes :",
              [j for i, j in x_train].count(0), "/",
              [j for i, j in y_train].count(0), "/",
              [j for i, j in w_train].count(0))
        print("-nb of slices :", len(x_train), "/", len(y_train), "/", len(w_train))
        
        print("Validation dataset")
        print("-nb of volumes :",
              [j for i, j in x_valid].count(0), "/",
              [j for i, j in y_valid].count(0), "/",
              [j for i, j in w_valid].count(0))
        print("-nb of slices :", len(x_valid), "/", len(y_valid), "/", len(w_valid))
        
        # not using the patch, only the full volume for test for now
        print("Testing dataset")
        print("-nb of volumes :",
              [j for i, j in x_test].count(0), "/",
              [j for i, j in y_test].count(0), "/",
              [j for i, j in w_test].count(0))
        print("-nb of slices :", len(x_test), "/", len(y_test), "/", len(w_test))
    
    # setup data augmentation
    aug = A.Compose([
        A.Rotate(limit=90, interpolation=1, border_mode=4, p=0.2),
        A.RandomRotate90((1, 2), p=0.2),
        A.Flip(0, p=0.2),
        A.Flip(1, p=0.2),
        A.Resize(input_shape[0], input_shape[1], interpolation=1, always_apply=True, p=1)
    ], additional_targets={"weight":"mask"}, p=1)

    # ready up the data generators
    train_gen = volume_slice_generator(x_train,
                                       y_train,
                                       weight=w_train,
                                       size=config['TRAINING']['batch2D'],
                                       augmentation=aug,
                                       shuffle=True)
    valid_gen = volume_slice_generator(x_valid,
                                       y_valid,
                                       weight=w_valid,  # not used for back propagation, but can be useful with callback to save best model
                                       size=config['TRAINING']['batch2D'],
                                       augmentation=None,
                                       shuffle=True)    # I'm kind of torn on putting shuffle on validation data, but otherwise, the extra patches are never used  
    test_gen  = volume_slice_generator(x_test,
                                       y_test,
                                       size=config['TRAINING']['batch2D'],
                                       augmentation=None,
                                       shuffle=False)

    if test_generator:
        print("Training gen info")
        print("-length :", len(train_gen))
        
        batch0 = train_gen[0]
        print("-batch output shape :", batch0[0].shape, "/", batch0[1].shape, end="")
        if len(batch0[0]) > 2:
            print(" /", batch0[2].shape)
        else:
            print()

        print("-batch visualization :")
        plt.figure(figsize=(10,20))
        
        # cool visuals which are a joy to debug
        nb_columns = len(batch0)
        for i in range(config['TRAINING']['batch2D']):
            for j, k in zip(range(1, nb_columns+1), ['x', 'y', 'w']):
                plt.subplot(config['TRAINING']['batch2D'], nb_columns, i*nb_columns+j)
                plt.title(k + " " + str(i), fontsize=12)
                plt.imshow(batch0[j-1][i], vmin=batch0[j-1].min(), vmax=batch0[j-1].max())
        
        plt.tight_layout()
        plt.show()

    # get unet model
    model = UNet(input_shape,
                 n_class=config['PREPROCESS']['n_cls'],
                 depth=depth,
                 n_feat=n_features,
                 backbone=backbone)
    
    total_steps = len(train_gen) * config['TRAINING']['epoch']
    schedule = tf.keras.optimizers.schedules.CosineDecay(LR, decay_steps=total_steps)
    optim = tf.keras.optimizers.Adam(learning_rate=schedule)

    if test_architecture:
        model.summary()

    if backbone:
        encoder_trainable(model, False)
        
        model.compile(optimizer=optim,
                      loss=bce_dice_loss,
                      metrics=metrics,
                      sample_weight_mode="temporal")

        # train the decoder a little before
        model.fit(train_gen,
                  validation_data=valid_gen,
                  steps_per_epoch=len(train_gen),
                  validation_steps=len(valid_gen),
                  batch_size=config['TRAINING']['batch2D'],
                  epochs=10)

        encoder_trainable(model, True)

    # compile model to add losses and metrics
    # (also needs to be called everytime we change trainable layers)
    model.compile(optimizer=optim,
                  loss=bce_dice_loss,
                  metrics=metrics,
                  sample_weight_mode="temporal")

    # training
    history = model.fit(train_gen,
                        validation_data=valid_gen,
                        steps_per_epoch=len(train_gen),
                        validation_steps=len(valid_gen),
                        batch_size=config['TRAINING']['batch2D'],
                        epochs=config['TRAINING']['epoch'],
                        verbose=1,
                        callbacks=[
                            model_checkpoint,
                            early_stop
                        ])

    # metrics display (acc, loss, etc.)
    loss_hist = history.history['loss']
    val_loss_hist = history.history['val_loss']
    epochs = range(1, len(loss_hist) + 1)
    plt.plot(epochs, loss_hist, 'y', label='Training loss')
    plt.plot(epochs, val_loss_hist, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(info_path + "_loss.png")
    plt.show() if test_prediction else plt.clf()
    
    dice_hist = history.history['dice']
    val_dice_hist = history.history['val_dice']

    plt.plot(epochs, dice_hist, 'y', label='Training Dice')
    plt.plot(epochs, val_dice_hist, 'r', label='Validation Dice')
    plt.title('Training and validation F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(info_path + "_dice.png")
    plt.show() if test_prediction else plt.clf()

    # test prediction
    best_model = tf.keras.models.load_model(model_path, compile=False)

    volume_patch_shape = [1] + input_shape[:-1]
    stride = [1] + [i // 2 for i in volume_patch_shape[1:]]

    tvol = tif.imread(test_volume)
    ttar = tif.imread(test_target).astype(float)  # can't be boolean for plt
    pred = volume_pred_from_img(best_model,
                                tvol,
                                stride=stride)
    ttar = resize(ttar,
                  output_shape=pred.shape,
                  order=0,
                  preserve_range=True,
                  anti_aliasing=False)
    
    if test_prediction:
        print("Prediction visualization :")
        
        plt.figure(figsize=(20,5))
        n = 5

        for i in range(n):
            plt.subplot(3,n,i+1)
            idx = tvol.shape[0] * i // n
            plt.title('slice ' + str(idx))
            plt.imshow(tvol[idx])
        
        for i in range(n):
            plt.subplot(3,n,i+1+n)
            idx = ttar.shape[0] * i // n
            plt.imshow(ttar[idx])

        for i in range(n):
            plt.subplot(3,n,i+1+2*n)
            idx = pred.shape[0] * i // n
            plt.imshow(pred[idx])
        
        plt.show()
    
    # show metrics results for test volume
    print("Test results :")
    y_true = tf.convert_to_tensor(ttar)
    y_pred = tf.convert_to_tensor(pred)
    for m in metrics:
        print("-", m.__name__, " : ", m(y_true, y_pred), sep="")

    # save test prediction if we want to check it out more
    tif.imwrite(info_path + "_pred.tif", pred)


if __name__ == "__main__":
    main()