# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tempfile
from tifffile import TiffFile, imread, imwrite
import carreno.nn.generators
import matplotlib.pyplot as plt
import albumentations as A
import volumentations as V
import matplotlib.pyplot as plt

verbose    = 0  # print to show progress
show_graph = 0  # plots to visualize data

def __label(x):
    # add color channels for classes
    lb = np.zeros([*x.shape, 3])
    lb[x < 2] = [1, 0, 0]
    lb[(x >= 2) & (x < 4)] = [0, 1, 0]
    lb[x >= 4] = [0, 0, 1]
    return lb

def main():
    # tmp dataset to test on
    print("{:.<25}... ".format("Generating tmp dataset "), end="") if verbose else ...
    
    y, z, x = np.meshgrid(range(6), range(3), range(6))
    
    if show_graph:
        plt.subplot(131)
        plt.title('z data ' + str(z.shape))
        plt.imshow(z[0])
        plt.subplot(132)
        plt.title('y data ' + str(y.shape))
        plt.imshow(y[0])
        plt.subplot(133)
        plt.title('x data ' + str(x.shape))
        plt.imshow(x[0])
        plt.show()
    
    input_dir = tempfile.TemporaryDirectory()
    input_paths = [input_dir.name + "/{}.tif".format(i) for i in ['x', 'y']]
    label_dir = tempfile.TemporaryDirectory()
    label_paths = [label_dir.name + "/{}.tif".format(i) for i in ['x', 'y']]
    weight_dir = tempfile.TemporaryDirectory()
    weight_paths = [weight_dir.name + "/{}.tif".format(i) for i in ['x', 'y']]
    
    # inputs and weights
    for path, volume in zip(input_paths + weight_paths, [x, y, z, z]):
        imwrite(path, volume, photometric='minisblack')
    
    # labels
    for path, volume in zip(label_paths, [x, y]):
        lb = __label(volume)
        imwrite(path, lb, photometric='rgb')  # its not RGB, but this will work for TiffFile pages
    
    print("done") if verbose else ...

    try:
        #################
        # 2D GENERATORS #
        #################

        print("2D gen :") if verbose else ...

        # volume slicing into images
        print(" - {:.<22}... ".format("volume slicing "), end="") if verbose else ...
        input_paths_w_slices = carreno.nn.generators.get_volumes_slices(input_paths)
        label_paths_w_slices = carreno.nn.generators.get_volumes_slices(label_paths)
        
        valid = len(input_paths) * x.shape[0]
        assert len(input_paths_w_slices) == valid, "number of volume slices should be {}, got {}".format(valid, len(input_paths_w_slices))
        print("done") if verbose else ...

        size = 1

        # img generator without augmentations
        print(" - {:.<22}... ".format("init gen w/ aug "), end="") if verbose else ...
        img_gen = carreno.nn.generators.volume_slice_generator(input_paths_w_slices,
                                                               label_paths_w_slices,
                                                               size=size,
                                                               augmentation=None,
                                                               noise=None,
                                                               shuffle=False,
                                                               weight=None)
        
        valid = len(input_paths_w_slices) // size
        assert len(img_gen) == valid, "number of batches should be {}, got {}".format(valid, len(img_gen))
        print("done") if verbose else ...

        # complete an epoch
        print(" - {:.<22}... ".format("iterate all batches "), end="") if verbose else ...
        for batch in img_gen:
            assert len(batch) == 2, "batch should have x and y, got {} values".format(len(batch))
            assert len(batch[0]) == size, "batch size should be {}, got {}".format(size, len(batch[0]))
            assert batch[0][0].ndim == 3, "number of dimensions should be 3 for an image, got shape of {}".format(batch[0][0].ndim)
        print("done") if verbose else ...
        
        if show_graph:
            plt.subplot(121)
            plt.title('batch x')
            plt.imshow(batch[0][0])
            plt.subplot(122)
            plt.title('batch y')
            plt.imshow(batch[1][0])
            plt.show()

        # data integrity after augmentation for last batch
        print(" - {:.<22}... ".format("data integrity w/ aug "), end="") if verbose else ...
        valid = y.shape[1:] + tuple([1])
        assert batch[0][0].shape == valid, "output shape for input should be {}, got shape of {}".format(valid, batch[0][0].shape)
        valid = y.shape[1:] + tuple([3])
        assert batch[1][0].shape == valid, "output shape for label should be {}, got shape of {}".format(valid, batch[0][0].shape)
        assert (np.squeeze(batch[0][0], axis=-1) == y).all(), "augmentation are off and yet batch input was transformed"
        assert (__label(np.squeeze(batch[0][0], axis=-1)) == batch[1][0]).all(), "augmentation are off and yet batch label was transformed"
        print("done") if verbose else ...
        
        # img generator with augmentations
        print(" - {:.<22}... ".format("init gen w aug "), end="") if verbose else ...
        flip2D = A.Compose([A.Flip(0, p=1)])
        img_gen = carreno.nn.generators.volume_slice_generator(input_paths_w_slices,
                                                               label_paths_w_slices,
                                                               size=size,
                                                               augmentation=flip2D,
                                                               noise=None,
                                                               shuffle=True,
                                                               weight=None)
        print("done") if verbose else ...

        # data integrity after augmentation
        print(" - {:.<22}... ".format("data integrity w aug "), end="") if verbose else ...
        batch = img_gen.__getitem__(0)

        if show_graph:
            plt.subplot(131)
            plt.title("transformed input")
            plt.imshow(batch[0][0])
            plt.subplot(132)
            plt.title("transformed label")
            plt.imshow(batch[1][0])
            plt.subplot(133)
            plt.title("transformed input to label")
            plt.imshow(__label(np.squeeze(batch[0][0], axis=-1)))
            plt.show()
        
        assert (__label(np.squeeze(batch[0][0], axis=-1)) == batch[1][0]).all(), \
            'flip transformation led to different results than expected'
        print("done") if verbose else ...

        """ TODO REVAMP WEIGHTS INTO PRECALCULATED PATCHES
        # weights 2D
        img_gen = carreno.nn.generators.volume_slice_generator(input_paths_w_slices,
                                                               label_paths_w_slices,
                                                               size=size,
                                                               augmentation=flip2D,
                                                               noise=None,
                                                               shuffle=True,
                                                               weight=weight_paths)
        """

        # noise 2D

        #################
        # 3D GENERATORS #
        #################

        print("3D gen :") if verbose else ...

        # volume generator without augmentations
        print(" - {:.<22}... ".format("init gen w/ aug "), end="") if verbose else ...
        vol_gen = carreno.nn.generators.volume_generator(input_paths,
                                                         label_paths,
                                                         size=size,
                                                         augmentation=None,
                                                         noise=None,
                                                         shuffle=False,
                                                         weight=None)
        valid = len(input_paths) // size
        assert len(vol_gen) == valid, "number of batches should be {}, got {}".format(valid, len(vol_gen))
        print("done") if verbose else ...

        # complete an epoch
        print(" - {:.<22}... ".format("iterate all batches "), end="") if verbose else ...
        for batch in vol_gen:
            assert len(batch) == 2, "batch should have x and y, got {} values".format(len(batch))
            assert len(batch[0]) == size, "batch size should be {}, got {}".format(size, len(batch[0]))
            assert batch[0][0].ndim == 4, "number of dimensions should be 3 for an image, got shape of {}".format(batch[0][0].shape)
        print("done") if verbose else ...

        # data integrity after augmentation for last batch
        print(" - {:.<22}... ".format("data integrity w/ aug "), end="") if verbose else ...
        valid = y.shape + tuple([1])
        assert batch[0][0].shape == valid, "output shape for input should be {}, got shape of {}".format(valid, batch[0][0].shape)
        valid = y.shape + tuple([3])
        assert batch[1][0].shape == valid, "output shape for label should be {}, got shape of {}".format(valid, batch[0][0].shape)
        assert (np.squeeze(batch[0][0], axis=-1) == y).all(), "augmentation are off and yet batch input was transformed"
        assert (__label(np.squeeze(batch[0][0], axis=-1)) == batch[1][0]).all(), "augmentation are off and yet batch label was transformed"
        print("done") if verbose else ...
        
        # volume generator with augmentations
        print(" - {:.<22}... ".format("init gen w aug "), end="") if verbose else ...
        flip3D = V.Compose([V.Flip(0, p=1)])
        vol_gen = carreno.nn.generators.volume_generator(input_paths,
                                                         label_paths,
                                                         size=size,
                                                         augmentation=flip3D,
                                                         noise=None,
                                                         shuffle=True,
                                                         weight=None)
        print("done") if verbose else ...

        # data integrity after augmentation
        print(" - {:.<22}... ".format("data integrity w aug "), end="") if verbose else ...
        batch = vol_gen.__getitem__(0)
        assert (__label(np.squeeze(batch[0][0], axis=-1)) == batch[1][0]).all(), \
            'flip transformation led to different results than expected'
        print("done") if verbose else ...

        # weights 3D

        # noise 3D

    except Exception as e:
        # cleanup dataset before raising exception
        input_dir.cleanup()
        label_dir.cleanup()
        weight_dir.cleanup()
        raise e
    else:
        # cleanup dataset
        input_dir.cleanup()
        label_dir.cleanup()
        weight_dir.cleanup()


if __name__ == "__main__":
    verbose    = 1
    show_graph = 1
    main()