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

def __flip3D(x):
    # horizontal flip to test transformations
    return np.flip(x, axis=2)


def test_gen(gen, n_input, x=None, y=None, w=None, size=1):
    """ TODO check noise
    test generator
    Parameters
    ----------
    gen : generator
        generator to iterate over and test
    n_input : int
        number of input
    x : array-like
        expected x output
    y : array-like
        expected y output
    w : array-like
        expected w output
    size : int
        batch size
    """
    using_weights = not w is None

    valid = n_input // size
    assert len(gen) == valid, "number of batches should be {}, got {}".format(valid, len(gen))
    
    # complete an epoch
    for batch in gen:
        if using_weights:
            assert len(batch) == 3, "batch should have x, y and w, got {} values".format(len(batch))
        else:
            assert len(batch) == 2, "batch should have x and y, got {} values".format(len(batch))

        assert len(batch[0]) == size, "batch size should be {}, got {}".format(size, len(batch[0]))
    
    # data integrity after augmentation for first batch
    batch = gen.__getitem__(0)
    for i, k, v in zip(range(3), ['x', 'y', 'w'], [x, y, w]):
        if not v is None:
            assert batch[i][0].shape == v.shape, "number of dimensions for {} should be {}, got shape of {}".format(k, v.shape, batch[i][0].shape)            
            assert (batch[i][0] == v).all(), "augmentations aren't giving the expecting result for {}".format(k)


def main():
    # tmp dataset to test on
    print("{:.<36}... ".format("Generating tmp dataset "), end="") if verbose else ...
    
    y, z, x = np.meshgrid(range(6), range(3), range(6))

    # create folders and file paths
    input_dir = tempfile.TemporaryDirectory()
    input_paths = [input_dir.name + "/{}.tif".format(i) for i in ['x', 'y']]
    
    label_dir = tempfile.TemporaryDirectory()
    label_paths = [label_dir.name + "/{}.tif".format(i) for i in ['x', 'y']]
    
    weight_dir = tempfile.TemporaryDirectory()
    weight_paths = [weight_dir.name + "/{}.tif".format(i) for i in ['x', 'y']]

    # inputs and weights
    for path, volume in zip(input_paths + weight_paths, [x, y] * 2):
        imwrite(path, volume, photometric='minisblack')
    
    # labels
    for path, volume in zip(label_paths, [x, y]):
        lb = __label(volume)
        imwrite(path, lb, photometric='rgb')
    print("done") if verbose else ...

    expected_x = np.expand_dims(x, axis=-1)
    expected_y = __label(x)
    expected_w = x

    try:
        #################
        # 2D GENERATORS #
        #################

        print("2D gen :") if verbose else ...

        # volume slicing into images
        print(" - {:.<33}... ".format("volume slicing "), end="") if verbose else ...
        input_paths_w_slices  = carreno.nn.generators.get_volumes_slices(input_paths)
        label_paths_w_slices  = carreno.nn.generators.get_volumes_slices(label_paths)
        weight_paths_w_slices = carreno.nn.generators.get_volumes_slices(weight_paths)
        
        valid = len(input_paths) * x.shape[0]
        assert len(input_paths_w_slices) == valid, "number of volume slices should be {}, got {}".format(valid, len(input_paths_w_slices))
        print("done") if verbose else ...

        size = 1

        # img generator without augmentations
        print(" - {:.<33}... ".format("init gen w/ aug "), end="") if verbose else ...
        img_gen = carreno.nn.generators.volume_slice_generator(input_paths_w_slices,
                                                               label_paths_w_slices,
                                                               size=size,
                                                               augmentation=None,
                                                               noise=None,
                                                               shuffle=False,
                                                               weight=None)
        
        test_gen(img_gen, len(input_paths_w_slices), expected_x[0], expected_y[0], w=None, size=1)
        print("done") if verbose else ...
        
        # img generator with augmentations
        print(" - {:.<33}... ".format("init gen w aug "), end="") if verbose else ...
        flip2D = A.Compose([A.HorizontalFlip(p=1)], additional_targets={'weight':'mask'}, p=1)
        img_gen = carreno.nn.generators.volume_slice_generator(input_paths_w_slices,
                                                               label_paths_w_slices,
                                                               weight=None,
                                                               size=size,
                                                               augmentation=flip2D,
                                                               noise=None,
                                                               shuffle=False)
        test_gen(img_gen, len(input_paths_w_slices), __flip3D(expected_x)[0], __flip3D(expected_y)[0], w=None, size=1)
        print("done") if verbose else ...

        # img generator with weights
        print(" - {:.<33}... ".format("init gen w weights "), end="") if verbose else ...
        img_gen = carreno.nn.generators.volume_slice_generator(input_paths_w_slices,
                                                               label_paths_w_slices,
                                                               weight=weight_paths_w_slices,
                                                               size=size,
                                                               augmentation=None,
                                                               noise=None,
                                                               shuffle=False)
        test_gen(img_gen, len(input_paths_w_slices), expected_x[0], expected_y[0], w=expected_w[0], size=1)
        print("done") if verbose else ...
        
        print(" - {:.<33}... ".format("init gen w aug n weights "), end="") if verbose else ...
        img_gen = carreno.nn.generators.volume_slice_generator(input_paths_w_slices,
                                                               label_paths_w_slices,
                                                               weight=weight_paths_w_slices,
                                                               size=size,
                                                               augmentation=flip2D,
                                                               noise=None,
                                                               shuffle=False)
        test_gen(img_gen, len(input_paths_w_slices), __flip3D(expected_x)[0], __flip3D(expected_y)[0], w=__flip3D(expected_w)[0], size=1)
        print("done") if verbose else ...

        # img generator with noise
        print(" - {:.<33}... ".format("init gen w noise "), end="") if verbose else ...
        grid_drop2D = A.Compose([A.GridDropout(0.5, holes_number_x=2, holes_number_y=2, p=1)], p=1)
        img_gen = carreno.nn.generators.volume_slice_generator(input_paths_w_slices,
                                                               label_paths_w_slices,
                                                               weight=None,
                                                               size=size,
                                                               augmentation=None,
                                                               noise=grid_drop2D,
                                                               shuffle=False)
        test_gen(img_gen, len(input_paths_w_slices), None, expected_y[0], w=None, size=1)
        print("done") if verbose else ...

        # img generator with everything
        print(" - {:.<33}... ".format("init gen w aug, weight and noise "), end="") if verbose else ...
        grid_drop2D = A.Compose([A.GridDropout(0.5, holes_number_x=2, holes_number_y=2, p=1)], p=1)
        img_gen = carreno.nn.generators.volume_slice_generator(input_paths_w_slices,
                                                               label_paths_w_slices,
                                                               weight=weight_paths_w_slices,
                                                               size=size,
                                                               augmentation=flip2D,
                                                               noise=grid_drop2D,
                                                               shuffle=False)
        test_gen(img_gen, len(input_paths_w_slices), None, __flip3D(expected_y)[0], w=__flip3D(expected_w)[0], size=1)
        print("done") if verbose else ...

        if show_graph:
            batch = img_gen.__getitem__(0)
            plt.subplot(231)
            plt.title('batch x')
            plt.imshow(batch[0][0])
            plt.subplot(232)
            plt.title('batch y')
            plt.imshow(batch[1][0])
            plt.subplot(233)
            plt.title('batch w')
            plt.imshow(batch[2][0])
            
            plt.subplot(234)
            plt.title('base x')
            plt.imshow(expected_x[0])
            plt.subplot(235)
            plt.title('base y')
            plt.imshow(expected_y[0])
            plt.subplot(236)
            plt.title('base w')
            plt.imshow(expected_w[0])

            plt.show()

        #################
        # 3D GENERATORS #
        #################

        print("3D gen :") if verbose else ...

        # volume generator without augmentations
        print(" - {:.<33}... ".format("init gen w/ aug "), end="") if verbose else ...
        vol_gen = carreno.nn.generators.volume_generator(input_paths,
                                                         label_paths,
                                                         size=size,
                                                         augmentation=None,
                                                         noise=None,
                                                         shuffle=False,
                                                         weight=None)
        test_gen(vol_gen, len(input_paths), expected_x, expected_y, w=None, size=1)
        print("done") if verbose else ...
        
        # volume generator with augmentations
        print(" - {:.<33}... ".format("init gen w aug "), end="") if verbose else ...
        flip3D = V.Compose([V.Flip(axis=2, always_apply=True, p=1)], targets=[['image'], ['mask', 'weight']], p=1)
        vol_gen = carreno.nn.generators.volume_generator(input_paths,
                                                         label_paths,
                                                         size=size,
                                                         augmentation=flip3D,
                                                         noise=None,
                                                         shuffle=False,
                                                         weight=None)
        test_gen(vol_gen, len(input_paths), __flip3D(expected_x), __flip3D(expected_y), w=None, size=1)
        print("done") if verbose else ...

        # weights 3D
        print(" - {:.<33}... ".format("init gen w weights "), end="") if verbose else ...
        vol_gen = carreno.nn.generators.volume_generator(input_paths,
                                                         label_paths,
                                                         weight=weight_paths,
                                                         size=size,
                                                         augmentation=None,
                                                         noise=None,
                                                         shuffle=False)
        test_gen(vol_gen, len(input_paths), expected_x, expected_y, w=expected_w, size=1)
        print("done") if verbose else ...
        
        print(" - {:.<33}... ".format("init gen w aug n weights "), end="") if verbose else ...
        vol_gen = carreno.nn.generators.volume_generator(input_paths,
                                                         label_paths,
                                                         weight=weight_paths,
                                                         size=size,
                                                         augmentation=flip3D,
                                                         noise=None,
                                                         shuffle=False)
        test_gen(vol_gen, len(input_paths), __flip3D(expected_x), __flip3D(expected_y), w=__flip3D(expected_w), size=1)
        print("done") if verbose else ...

        # noise 3D
        print(" - {:.<33}... ".format("init gen w noise "), end="") if verbose else ...
        grid_drop2D = V.Compose([V.GridDropout(0.5, holes_number_x=3, holes_number_y=1, holes_number_z=3, p=1)], p=1)
        vol_gen = carreno.nn.generators.volume_generator(input_paths,
                                                         label_paths,
                                                         weight=None,
                                                         size=size,
                                                         augmentation=None,
                                                         noise=grid_drop2D,
                                                         shuffle=False)
        test_gen(vol_gen, len(input_paths), None, expected_y, w=None, size=1)
        print("done") if verbose else ...

        # with everything
        print(" - {:.<33}... ".format("init gen w aug, weight and noise "), end="") if verbose else ...
        grid_drop2D = V.Compose([V.GridDropout(0.5, holes_number_x=2, holes_number_y=1, holes_number_z=2, p=1)], p=1)
        vol_gen = carreno.nn.generators.volume_generator(input_paths,
                                                         label_paths,
                                                         weight=weight_paths,
                                                         size=size,
                                                         augmentation=flip3D,
                                                         noise=grid_drop2D,
                                                         shuffle=False)
        test_gen(vol_gen, len(input_paths), None, __flip3D(expected_y), w=__flip3D(expected_w), size=1)
        print("done") if verbose else ...

        if show_graph:
            batch = vol_gen.__getitem__(0)
            plt.subplot(231)
            plt.title('batch x')
            plt.imshow(batch[0][0][0])
            plt.subplot(232)
            plt.title('batch y')
            plt.imshow(batch[1][0][0])
            plt.subplot(233)
            plt.title('batch w')
            plt.imshow(batch[2][0][0])

            plt.subplot(234)
            plt.title('base x')
            plt.imshow(expected_x[0])
            plt.subplot(235)
            plt.title('base y')
            plt.imshow(expected_y[0])
            plt.subplot(236)
            plt.title('base w')
            plt.imshow(expected_w[0])

            plt.show()

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