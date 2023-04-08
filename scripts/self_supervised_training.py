# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import albumentations as A
import volumentations as V
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# local imports
import utils
import carreno.nn.callbacks as cb
import carreno.nn.metrics as mtc
from carreno.nn.unet import encoder_trainable
from carreno.nn.generators import get_volumes_slices, volume_generator, volume_slice_generator

# adjustable settings
train_encoder  = False
LR             = 0.00001
loss           = mtc.dice_loss  # tf.keras.losses.MeanSquaredError() # TODO would probably get much better results with GAN loss
inp_model_name = "untrn_unet2D-4-64-0.3-relu-VGG16.h5"
out_model_name = "slfspv_{}-{}.h5".format(LR, inp_model_name.rsplit('.', 1)[0])


def main(verbose=0):
    config = utils.get_config()

    # load model
    path_to_unet = os.path.join(config['TRAINING']['output'], inp_model_name)
    model = tf.keras.models.load_model(path_to_unet, compile=False)
    
    # find if model is 2D or 3D
    is_2D = len(model.layers[0].input.shape) == 4  # 5 if 3D
    print("Model is {}D".format(2 if is_2D else 3)) if verbose else ...

    # split dataset
    vtype = 'unlabeled'
    dataset = utils.split_dataset(vtype,
                                  [vtype],
                                  valid=0.2 * 2,  # doubled from 0.2 to 0.4 because there's no ctrl
                                  test =0.2 * 2,  # meaning we only take half (see implementation in utils)
                                  shuffle=True)
    
    x_train, x_valid, y_train, y_valid = train_test_split(x_data,
                                                          y_data,
                                                          test_size=0.4,
                                                          random_state=6)
    x_valid, x_test, y_valid, y_test   = train_test_split(x_valid,
                                                          y_valid,
                                                          test_size=0.5,
                                                          random_state=9)
    
    if verbose:
        for i, txt in zip(list(range(len(dataset))), ['training', 'validation', 'testing']):
            print("Number of patches for {} :".format(txt), len(dataset[i][vtype]))
    
    # make data generator for model inputs
    train_gen, valid_gen, test_gen = [None] * 3
    if is_2D:
        # slice up volumes over axis 0
        slice_dataset = []
        for set in dataset:
            slice_dataset.append(get_volumes_slices(set[vtype]))
        
        if verbose:
            for i, txt in zip(list(range(len(slice_dataset))), ['training', 'validation', 'testing']):
                print("Number of slices for {} :".format(txt), len(slice_dataset[i]))
        
        # augmentation
        aug = None  # we already have plenty of slices
        noise = A.Compose([
            A.GridDropout(0.5, unit_size_min=5, unit_size_max=15,
                          holes_number_x=5, holes_number_y=5,
                          random_offset=True, p=1)
        ])

        gens = []
        for i in range(len(slice_dataset)):
            gen = volume_slice_generator(vol=slice_dataset[i],
                                         label=slice_dataset[i],
                                         weight=slice_dataset[i] if i < 2 else None,  # no weights for tests
                                         size=config['TRAINING']['batch2D'],
                                         augmentation=aug,
                                         noise=noise,
                                         shuffle=True,
                                         nb_color_ch=3)
            gens.append(gen)

        train_gen, valid_gen, test_gen = gens
    else:
        # augmentation
        aug = None  # we already have plenty of patches
        noise = V.Compose([
            V.GridDropout(0.5, unit_size_min=5, unit_size_max=15,
                          holes_number_x=5, holes_number_y=5, holes_number_z=2,
                          random_offset=True, p=1)
        ])

        gens = []
        for i in range(len(dataset)):
            gen = volume_generator(vol=dataset[i][vtype],
                                   label=dataset[i][vtype],
                                   weight=dataset[i][vtype] if i < 2 else None,  # no weights for tests
                                   size=config['TRAINING']['batch3D'],
                                   augmentation=aug,
                                   noise=noise,
                                   shuffle=True,
                                   nb_color_ch=3)
            gens.append(gen)
        
        train_gen, valid_gen, test_gen = gens
    
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
        plt.figure(figsize=(5,7), dpi=150)
        
        # cool visuals which are a joy to debug
        nb_columns = len(batch0)
        btc_size   = config['TRAINING']['batch2D'] if is_2D else config['TRAINING']['batch3D']
        nb_lines   = min(btc_size, 2)
        for i in range(nb_lines):
            for j, k in zip(range(1, nb_columns+1), ['x', 'y', 'w']):
                plt.subplot(nb_lines, nb_columns, i*nb_columns+j)
                plt.title(k + " " + str(i), fontsize=12)
                plt.imshow(batch0[j-1][i], vmin=batch0[j-1].min(), vmax=batch0[j-1].max())
        
        plt.tight_layout()
        plt.show()

    # freeze encoder if needed
    encoder_trainable(model, train_encoder)

    # set callbacks
    metrics = [mtc.dice_score(smooth=1.)]
    
    early_stop = cb.early_stop(metric='val_dice',
                               mode='max',
                               patience=config['TRAINING']['patience'])
    
    out_model_path = os.path.join(config['TRAINING']['output'], out_model_name)
    model_checkpoint = cb.model_checkpoint(out_model_path,
                                           metric='val_dice',
                                           mode='max')

    total_steps = len(train_gen) * config['TRAINING']['epoch']
    schedule = tf.keras.optimizers.schedules.CosineDecay(LR, decay_steps=total_steps)
    optim = tf.keras.optimizers.Adam(learning_rate=schedule)

    model.compile(optimizer=optim,
                  loss=loss,
                  metrics=metrics,
                  sample_weight_mode="temporal")

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
    graph_path = out_model_path.rsplit(".", 1)[0]

    loss_hist = history.history['loss']
    val_loss_hist = history.history['val_loss']
    epochs = range(1, len(loss_hist) + 1)
    plt.plot(epochs, loss_hist, 'y', label='Training loss')
    plt.plot(epochs, val_loss_hist, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(graph_path + "_loss.png")
    plt.show() if verbose else plt.clf()
    
    dice_hist = history.history['dice']
    val_dice_hist = history.history['val_dice']

    plt.plot(epochs, dice_hist, 'y', label='Training Dice')
    plt.plot(epochs, val_dice_hist, 'r', label='Validation Dice')
    plt.title('Training and validation F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(graph_path + "_dice.png")
    plt.show() if verbose else plt.clf()

    # evaluate model
    best_version = tf.keras.models.load_model(out_model_path,
                                              custom_objects={"dice": metrics[0],
                                                              "dice_loss": loss})
    best_version.evaluate(test_gen,
                          verbose=1)


if __name__ == "__main__":
    main(verbose=1)