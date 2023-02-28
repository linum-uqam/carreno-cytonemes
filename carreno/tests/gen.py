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

verbose = 0

def __label(x):
    # add color channels for classes
    lb = np.zeros([*x.shape, 2])
    lb[x < 2] = [1, 0]
    lb[x >= 2] = [0, 1]
    return lb

def main():
    print("Generating tmp dataset ... ", end="") if verbose else ...
    
    y, z, x = np.meshgrid(range(6), range(3), range(6))
    """
    print(z.shape, y.shape, x.shape)
    plt.subplot(131)
    plt.imshow(z[0])
    plt.subplot(132)
    plt.imshow(y[0])
    plt.subplot(133)
    plt.imshow(x[0])
    plt.show()
    """
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
        imwrite(path, __label(volume), photometric='rgb')  # its not RGB, but this will work for TiffFile pages
    
    print("done") if verbose else ...

    try:
        input_paths_w_slices = carreno.nn.generators.get_volumes_slices(input_paths)
        label_paths_w_slices = carreno.nn.generators.get_volumes_slices(label_paths)
        
        valid = len(input_paths) * x.shape[0]
        assert len(input_paths_w_slices) == valid, "number of volume slices should be {}, got {}".format(valid, len(input_paths_w_slices))

        size = 1
        img_gen = carreno.nn.generators.volume_slice_generator(input_paths_w_slices,
                                                               label_paths_w_slices,
                                                               size=size,
                                                               augmentation=None,
                                                               noise=None,
                                                               shuffle=False,
                                                               weight=None)
        
        valid = len(input_paths_w_slices) // size
        assert len(img_gen) == valid, "number of batches should be {}, got {}".format(valid, len(img_gen))

        for batch in img_gen:
            assert len(batch) == 2, "batch should have x and y, got {} values".format(len(batch))
            assert len(batch[0]) == size, "batch size should be {}, got {}".format(size, len(batch[0]))
            assert batch[0][0].ndim == 3, "number of dimensions should be 3 for an image, got shape of {}".format(batch[0][0].shape)
        
        assert (np.squeeze(batch[0][0], axis=-1) == x).all() or (__label(np.squeeze(batch[0][0], axis=-1)) == batch[1][0]).all(), \
            'augmentation are off and yet batch was transformed'

        # augmentation 2D
        flip2D = A.Compose([A.Flip(0, p=1)])
        img_gen = carreno.nn.generators.volume_slice_generator(input_paths_w_slices,
                                                               label_paths_w_slices,
                                                               size=size,
                                                               augmentation=flip2D,
                                                               noise=None,
                                                               shuffle=True,
                                                               weight=None)
        
        batch = img_gen.__getitem__(0)
        """
        plt.subplot(131)
        plt.imshow(batch[0][0][0])
        plt.subplot(132)
        plt.imshow(batch[1][0][0])
        plt.subplot(133)
        plt.imshow(__label(np.squeeze(batch[0][0], axis=-1))[0])
        plt.show()
        """
        assert (__label(np.squeeze(batch[0][0], axis=-1)) == batch[1][0]).all(), \
            'flip transformation led to different results than expected'

        """
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

        # add color channel 2D

        # creation 3D

        # iteration 3D

        # augmentation 3D

        # weights 3D

        # noise 3D

        # add color channel 2D
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
    verbose = 1
    main()