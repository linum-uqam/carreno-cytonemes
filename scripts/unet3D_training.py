# -*- coding: utf-8 -*-
import numpy as np
import tifffile as tif
import os
import tensorflow as tf
import volumentations as V
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from skimage.transform import resize

from carreno.nn.unet import UNet, unet2D_to_unet3D
import carreno.nn.callbacks as cb
from carreno.nn.generators import volume_generator
from carreno.nn.metrics import dice_score, bce_dice_loss
from carreno.processing.patchify import volume_pred_from_vol

# validate dataset is present
data_folder = "data"  # folder where downloads and dataset will be put
dataset_folder = data_folder +    "/dataset"
output_folder  = data_folder +    "/output"
input_folder   = dataset_folder + "/input_p"
target_folder  = dataset_folder + "/target_p"
test_volume    = dataset_folder + "/input/slik3.tif"
test_target    = dataset_folder + "/target/slik3.tif"
model_path     = output_folder +  "/model/unet3D.h5"
unet2d_model   = output_folder +  "/model/unet2D.h5"
info_path      = output_folder +  "/" + Path(model_path).name.split('.')[0]
nb_class = 3
batch_size = 8
nb_epochs = 50
input_shape = [64, 64, 64, 1]
class_weights = "balanced"

# visualization
test_split          = 1  # show if data split info
test_generator      = 0  # show training gen output
test_architecture   = 1  # show model summary
test_prediction     = 0  # show a few prediction slices

def main():
    # split data between training, validation and test
    x_data = []
    x_test = []
    test_vol_name = Path(test_volume).name.split('.')[0]
    for f in os.listdir(input_folder):
        if test_vol_name in f.split('_')[0]:
            x_test.append(input_folder + "/" + f)
        else:
            x_data.append(input_folder + "/" + f)
        
    y_data = []
    y_test = []
    for f in os.listdir(target_folder):
        if test_vol_name in f.split('_')[0]:
            y_test.append(target_folder + "/" + f)
        else:
            y_data.append(target_folder + "/" + f)            

    x_train, x_valid, y_train, y_valid = train_test_split(x_data,
                                                          y_data,
                                                          test_size=0.2,
                                                          random_state=6)

    if test_split:
        print("Training dataset")
        print("-nb of volumes :",
              len(x_train), "/",
              len(y_train))
        
        print("Validation dataset")
        print("-nb of volumes :",
              len(x_valid), "/",
              len(y_valid))
        
        # not using the patch, only the full volume for test for now
        print("Testing dataset")
        print("-nb of volumes :",
              len(x_test), "/",
              len(y_test))
        
    # setup data augmentation
    aug = V.Compose([
        V.RandomRotate90((1, 2), p=0.2),
        V.Flip(0, p=0.2),
        V.Flip(1, p=0.2),
        V.Rotate((-90, 90), (0,0), (0, 0), border_mode='reflect', p=0.2),
        V.ElasticTransformPseudo2D(alpha=40, sigma=10, alpha_affine=1, p=0.2),
        #V.ElasticTransform((0, 0.10), interpolation=1, border_mode='reflect', p=0.2),
        #V.Resize(input_shape[:-1], interpolation=1, always_apply=True, p=1)  # (don't include color channel in shape) TODO doesn't work animore
    ], p=1)

    # ready up the data generators
    train_gen = volume_generator(x_train,
                                 y_train,
                                 batch_size,
                                 augmentation=aug,
                                 shuffle=True,
                                 weight=class_weights)
    valid_gen = volume_generator(x_valid,
                                 y_valid,
                                 batch_size,
                                 augmentation=None,
                                 shuffle=False)
    test_gen  = volume_generator(x_test,
                                 y_test,
                                 batch_size,
                                 augmentation=None,
                                 shuffle=False)
    
    train_gen.on_epoch_end()  # shuffle

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
        z = 15
        nb_columns = len(batch0)
        for i in range(batch_size):
            for j, k in zip(range(1, nb_columns+1), ['x', 'y', 'w']):
                plt.subplot(batch_size, nb_columns, i*nb_columns+j)
                plt.title(k + " " + str(i), fontsize=12)
                plt.imshow(batch0[j-1][i][z], vmin=batch0[j-1].min(), vmax=batch0[j-1].max())
        
        plt.tight_layout()
        plt.show()

    # get unet model
    model = None
    if unet2d_model is None:
        model = UNet(input_shape, nb_class)
    else:
        # transfer learning
        unet2D = tf.keras.models.load_model(unet2d_model, compile=False)
        model = unet2D_to_unet3D(unet2D,
                                 shape=input_shape)

    if test_architecture:
        model.summary()

    # parameters and hyperparameters
    LR = 0.0001
    optim = tf.keras.optimizers.Adam(LR)

    # metrics
    metrics = [dice_score(smooth=1.)]

    # callbacks
    early_stop = cb.early_stop(metric='val_dice',
                               mode='max',
                               patience=5)
    model_checkpoint = cb.model_checkpoint(model_path,
                                           metric='val_dice',
                                           mode='max')
    cm_history = cb.confusion_matrix_history(train_generator=train_gen,
                                             validation_generator=valid_gen,
                                             n_class=nb_class)

    # compile model
    model.compile(optimizer=optim,
                  loss=bce_dice_loss,
                  metrics=metrics,
                  sample_weight_mode="temporal")

    # training
    history = model.fit(train_gen,
                        validation_data=valid_gen,
                        steps_per_epoch=len(train_gen),
                        validation_steps=len(valid_gen),
                        batch_size=batch_size,
                        epochs=nb_epochs,
                        verbose=1,
                        callbacks=[
                            model_checkpoint,
                            early_stop,
                            #cm_history
                        ])

    # in case we didn't use cm callback to save time
    if len(cm_history.tcm) > 0:
        # save confusion matrix history in callback
        np.save(info_path + "_tcm.npy", np.stack(cm_history.tcm, axis=0))  # training cm
        np.save(info_path + "_vcm.npy", np.stack(cm_history.vcm, axis=0))  # validation cm
    
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
    if test_prediction:
        plt.show()
    else:
        plt.clf()

    dice_hist = history.history['dice']
    val_dice_hist = history.history['val_dice']

    plt.plot(epochs, dice_hist, 'y', label='Training Dice')
    plt.plot(epochs, val_dice_hist, 'r', label='Validation Dice')
    plt.title('Training and validation F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(info_path + "_dice.png")
    if test_prediction:
        plt.show()
    else:
        plt.clf()

    # test prediction
    best_model = tf.keras.models.load_model(model_path, compile=False)

    volume_patch_shape = input_shape[:-1]
    stride = [i // 2 for i in volume_patch_shape]

    tvol = tif.imread(test_volume)
    ttar = tif.imread(test_target).astype(float)  # can't be boolean for plt
    pred = volume_pred_from_vol(best_model,
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