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
from sklearn.metrics import confusion_matrix

import carreno.nn.unet as unet
import carreno.nn.callbacks as ccb
from carreno.nn.generators import volume_slice_generator
from carreno.nn.metrics import dice_score, bce_dice_loss
from carreno.processing.patchify import volume_pred_from_img

# validate dataset is present
data_folder = "data"  # folder where downloads and dataset will be put
dataset_name = "dataset"
input_folder  = data_folder + "/" + dataset_name + "/input_p"
target_folder = data_folder + "/" + dataset_name + "/target_p"
model_path = data_folder + "/model/test.h5"
test_volume = data_folder + "/" + dataset_name + "/input/9.tif"
test_target = data_folder + "/" + dataset_name + "/target/9.tif"
test_save = data_folder + "/output/pred_unet2d.tif"
nb_class = 3
batch_size = 16
input_shape = [64, 64, 1]
class_weights = "balanced"

# visualization
test_split          = True  # show if data split info
test_generator      = True  # show training gen output
test_architecture   = True  # show model summary
test_prediction     = True  # show a few prediction slices

def get_volumes_slices(paths):
    """
    Get all slice index for all the volumes in list of paths
    Parameters
    ----------
    paths : [str]
        Paths to volumes to slice up
    Returns
    -------
    slices : [[str, int]]
        list of list containing volume names and slice indexes
    """
    slices = []

    for path in paths:
        tif_file = tif.TiffFile(path)
        for i in range(len(tif_file.pages)):
            slices.append([path, i])
    
    return slices


class Fit_w_confusion_matrix(tf.keras.Model):
    def __init__(self, model):
        super(Fit_w_confusion_matrix, self).__init__()
        self.model = model
        self.tcm = []
        self.vcm = []
    
    def train_step(self, data):
        x = data[0]
        y = data[1]
        w = None
        if len(data) > 2:
            w = tf.concat(concat_dims=3, values=[data[2]] * nb_class)
        
        # forward propagation
        with tf.GradientTape as tape:
            y_pred = self.model(x, training=True)
            loss = 1e7
            if w is None:
                loss = self.compiled_loss(y, y_pred)
            else:
                loss = self.compiled_loss(y * w, y_pred * w)
            
        # back propagation
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        self.optimizer.apply_gradients(zip(gradients, training_vars))
        self.compiled_metrics.update_state(y, y_pred)
        self.tcm.append(confusion_matrix(y, y_pred))

        return {"loss" : loss}.update( {m.name : m.result() for m in self.metrics} )
    
    def test_step(self, data):
        x, y = data

        y_pred = self.model(x, training=False)
        loss = self.compiled_loss(y, y_pred)
        
        self.compiled_metrics.update_state(y, y_pred)
        self.vcm.append(confusion_matrix(y, y_pred))

        return {"val_loss" : loss}.update( {m.name : m.result() for m in self.metrics} )


def main():
    # split data between training, validation and test
    x_files = []
    for f in os.listdir(input_folder):
        x_files.append(input_folder + "/" + f)
        
    y_files = []
    for f in os.listdir(target_folder):
        y_files.append(target_folder + "/" + f)

    x_train, x_valid, y_train, y_valid = train_test_split(x_files,
                                                        y_files,
                                                        test_size=0.2,
                                                        random_state=6)

    # slice up volumes
    x_train = get_volumes_slices(x_train)
    y_train = get_volumes_slices(y_train)
    x_valid = get_volumes_slices(x_valid)
    y_valid = get_volumes_slices(y_valid)

    if test_split:
        print("Training dataset")
        print("-nb of volumes :",
            [j for i, j in x_train].count(0), "/",
            [j for i, j in y_train].count(0))
        print("-nb of slices :", len(x_train), "/", len(y_train))
        
        print("Validation dataset")
        print("-nb of volumes :",
            [j for i, j in x_valid].count(0), "/",
            [j for i, j in y_valid].count(0))
        print("-nb of slices :", len(x_valid), "/", len(y_valid))
        
        """ no test data since we already a full volume saved for it later
        print("Testing dataset")
        print("-nb of volumes :",
            [j for i, j in x_test].count(0), "/",
            [j for i, j in y_test].count(0))
        print("-nb of slices :", len(x_test), "/", len(y_test))
        """

    # setup data augmentation
    aug = A.Compose([
        A.Rotate(limit=90, interpolation=1, border_mode=4, p=0.2),
        A.RandomRotate90((1, 2), p=0.2),
        A.Flip(0, p=0.2),
        A.Flip(1, p=0.2),
        A.Resize(input_shape[0], input_shape[1], interpolation=1, always_apply=True, p=1)
    ], p=1)

    # ready up the data generators
    train_gen = volume_slice_generator(x_train,
                                    y_train,
                                    batch_size,
                                    augmentation=aug,
                                    shuffle=True,
                                    weight=class_weights)
    valid_gen = volume_slice_generator(x_valid,
                                    y_valid,
                                    batch_size,
                                    augmentation=None,
                                    shuffle=False)
    """
    test_gen  = volume_slice_generator(x_test,
                                    y_test,
                                    batch_size,
                                    augmentation=None,
                                    shuffle=False)
    """
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
        nb_columns = len(batch0)
        for i in range(batch_size):
            for j, k in zip(range(1, nb_columns+1), ['x', 'y', 'w']):
                plt.subplot(batch_size, nb_columns, i*nb_columns+j)
                plt.title(k + " " + str(i), fontsize=12)
                plt.imshow(batch0[j-1][i], vmin=batch0[j-1].min(), vmax=batch0[j-1].max())
        
        plt.tight_layout()
        plt.show()

    # get unet model
    model = unet.unet2D(input_shape, nb_class)

    if test_architecture:
        model.summary()

    # parameters and hyperparameters
    LR = 0.0001
    optim = tf.keras.optimizers.Adam(LR)

    # metrics
    metrics = [dice_score(smooth=1.)]

    # callbacks
    early_stop = ccb.early_stop(metric='val_dice',
                                mode='max',
                                patience=5)
    model_checkpoint = ccb.model_checkpoint(model_path,
                                            metric='val_dice',
                                            mode='max')

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
                        epochs=50,
                        verbose=1,
                        callbacks=[model_checkpoint, early_stop])

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
    plt.show()

    dice_hist = history.history['dice']
    val_dice_hist = history.history['val_dice']

    plt.plot(epochs, dice_hist, 'y', label='Training Dice')
    plt.plot(epochs, val_dice_hist, 'r', label='Validation Dice')
    plt.title('Training and validation F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()

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
    folder = os.path.dirname(test_save)
    Path(folder).mkdir(parents=True, exist_ok=True)
    tif.imwrite(test_save, pred)


if __name__ == "__main__":
    main()