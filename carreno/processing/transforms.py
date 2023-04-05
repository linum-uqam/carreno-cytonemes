import numpy as np
import tifffile as tif
import random as rnd
import scipy.ndimage as nd
import tensorflow as tf

import carreno.utils.array

class Transform:
    # Inspired by Volumentations
    # https://github.com/ZFTurbo/volumentations/blob/master/volumentations/core/transforms_interface.py
    def __init__(self, p=1):
        """
        Transformation parent class. Has a probability of being applied.
        Parameters
        ----------
        p : float
            Probability of applying transformation between 0 and 1
        """
        assert 0 <= p <= 1, "Probability should be between 0 and 1, got {}".format(p)
        self.p = p

    def __call__(self, x, y=None, w=None):
        """
        Apply transformation in `apply` method to data if probabilities are met.
        Parameters
        ----------
        x : array-like
            input
        y : array-like, None
            target
        w : array-like, None
            weight
        Returns
        -------
        x : array-like
            Possibly transformed input
        y : array-like, None
            Possibly transformed target
        w : array-like, None
            Possibly transformed weight
        """
        xyw = x, y, w
        if self.p == 1 or rnd.random() < self.p:  # if we put rnd <= p, p=0 could be applied
            xyw = self.apply(*xyw)                # if we put rnd <  p, p=1 could be not applied
        return xyw
    
    def apply(self, x, y=None, w=None):
        raise NotImplementedError
    

class Compose(Transform):
    def __init__(self, transforms=[], **kwargs):
        """
        Combination of transformations
        Parameters
        ----------
        transforms : list
            List of transformations
        """
        super().__init__(**kwargs)
        self.transforms = transforms

    def apply(self, x, y=None, w=None):
        """
        Apply all transformations in order to data.
        Parameters
        ----------
        x : array-like
            input
        y : array-like, None
            target
        w : array-like, None
            weight
        Returns
        -------
        x : array-like
            Transformed input
        y : array-like, None
            Transformed target
        w : array-like, None
            Transformed weight
        """
        xyw = [x.copy(),
               None if y is None else y.copy(),
               None if w is None else w.copy()]
        for tf in self.transforms:
            xyw = tf(*xyw)
        return xyw


class Read(Transform):
    def __init__(self, **kwargs):
        """
        Read tif files into ndarray.
        """
        super().__init__(**kwargs)

    def apply(self, x, y=None, w=None):
        """
        Read tif files into ndarray.
        Parameters
        ----------
        x : Path, str
            Path to input
        y : Path, str, None
            Path to target
        w : Path, str, None
            Path to weight
        Returns
        -------
        x : array-like
            Input array
        y : array-like, None
            Target array
        w : array-like, None
            Weight array
        """
        return [tif.imread(i) if i else None for i in (x, y, w)]


class PadResize(Transform):
    def __init__(self, shape, mode="constant", cval=0, **kwargs):
        """
        Pad data to fit a minimum shape.
        Parameters
        ----------
        shape : array-like
            Minimum dimensions for data
        mode : str, function
            Padding type.
            Refer to mode param at https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        cval : __
            Value for constant padding.
            Refer to constant_values param at https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        """
        super().__init__(**kwargs)
        self.shape = np.array(shape)
        self.mode  = mode
        self.cval = cval

    def apply(self, x, y=None, w=None):
        """
        Pad data to fit a minimum shape.
        Parameters
        ----------
        x : array-like
            Input
        y : array-like, None
            Target
        w : array-like, None
            Weight
        Returns
        -------
        x : array-like
            Padded input
        y : array-like, None
            Padded target
        w : array-like, None
            Padded weight
        """
        ndim_to_transform = self.shape.shape[0]
        if not y is None:
            assert x.shape[:ndim_to_transform] == y.shape[:ndim_to_transform], \
                "Data shape conflict, got x={}, y={}".format(x.shape[:ndim_to_transform], y.shape[:ndim_to_transform])
        if not w is None:
            assert x.shape[:ndim_to_transform] == w.shape[:ndim_to_transform], \
                "Data shape conflict, got x={}, w={}".format(x.shape[:ndim_to_transform], w.shape[:ndim_to_transform])
        padding = [[0,0]] * x.ndim
        for i in range(len(self.shape)):
            diff = x.shape[i] - self.shape[i]
            if diff < 0:
                split = abs(diff) / 2
                padding[i] = [int(i) for i in [np.floor(split), np.ceil(split)]]
        return [None if i is None else np.pad(i, pad_width=padding, mode=self.mode, constant_values=self.cval) for i in (x,y,w)]


class Sample(Transform):
    def __init__(self, shape, **kwargs):
        """
        Random crop/sample data.
        Parameters
        ----------
        shape : list
            Shape of sample
        """
        super().__init__(**kwargs)
        self.shape = np.array(shape)
    
    def apply(self, x, y=None, w=None):
        """
        Random crop/sample of data.
        Parameters
        ----------
        x : array-like
            Input
        y : array-like, None
            Target
        w : array-like, None
            Weight
        Returns
        -------
        x : array-like
            Cropped input
        y : array-like, None
            Cropped target
        w : array-like, None
            Cropped weight
        """
        ndim_to_transform = self.shape.shape[0]
        if not y is None:
            assert x.shape[:ndim_to_transform] == y.shape[:ndim_to_transform], \
                "Data shape conflict, got x={}, y={}".format(x.shape[:ndim_to_transform], y.shape[:ndim_to_transform])
        if not w is None:
            assert x.shape[:ndim_to_transform] == w.shape[:ndim_to_transform], \
                "Data shape conflict, got x={}, w={}".format(x.shape[:ndim_to_transform], w.shape[:ndim_to_transform])
        assert (self.shape <= x.shape).all(), "Sample shape is to big for data, got x={}, sample={}".format(x.shape, self.shape)
        slices = []
        for i in range(len(self.shape)):
            sample_len = self.shape[i]
            options_range = [0, x.shape[i] - sample_len]
            chosen_one = rnd.randint(*options_range)
            slices.append(slice(chosen_one, chosen_one + sample_len))
        slices = tuple(slices)
        return [None if i is None else i[slices] for i in (x,y,w)]


class Normalize(Transform):
    def __init__(self, min=0, max=1, **kwargs):
        """
        Normalize data between min and max.
        Parameters
        ----------
        min : float
            Minimum value
        max : float
            Maximum value
        """
        super().__init__(**kwargs)
        self.min = min
        self.max = max
    
    def apply(self, x, y=None, w=None):
        """
        Normalize data between min and max.
        Parameters
        ----------
        x : array-like
            Input
        y : array-like, None
            Target
        w : array-like, None
            Weight
        Returns
        -------
        x : array-like
            Normalized input
        y : array-like, None
            Normalized target
        w : array-like, None
            Normalized weight
        """
        return carreno.utils.array.normalize(x, self.min, self.max), y, w


class Standardize(Transform):
    def __init__(self, **kwargs):
        """
        Standardize input data.
        """
        super().__init__(**kwargs)

    def apply(self, x, y=None, w=None):
        """
        Standardize input data.
        Parameters
        ----------
        x : array-like
            Input
        y : array-like, None
            Target
        w : array-like, None
            Weight
        Returns
        -------
        x : array-like
            Standardized input
        y : array-like, None
            Standardized target
        w : array-like, None
            Standardized weight
        """
        return carreno.utils.array.standardize(x), y, w


class Squeeze(Transform):
    def __init__(self, axis=-1, **kwargs):
        """
        Squeeze axis for data.
        Parameters
        ----------
        axis : int
            ndarray axis to squeeze
        """
        super().__init__(**kwargs)
        self.axis = axis
    
    def apply(self, x, y=None, w=None):
        """
        Standardize input data.
        Parameters
        ----------
        x : array-like
            Input
        y : array-like, None
            Target
        w : array-like, None
            Weight
        Returns
        -------
        x : array-like
            Squeezed input
        y : array-like, None
            Squeezed target
        w : array-like, None
            Squeezed weight
        """
        return [None if i is None else np.squeeze(i, axis=self.axis) for i in (x,y,w)]
    

class Stack(Transform):
    def __init__(self, axis=-1, n=1, **kwargs):
        """
        Stacked input data.
        Parameters
        ----------
        axis : int
            ndarray axis to stack
        n : int
            Number of time to stack axis
        """
        super().__init__(**kwargs)
        self.axis = axis
        self.n    = n
    
    def apply(self, x, y=None, w=None):
        """
        Stacked input data.
        Parameters
        ----------
        x : array-like
            Input
        y : array-like, None
            Target
        w : array-like, None
            Weight
        Returns
        -------
        x : array-like
            Stacked input
        y : array-like, None
            Stacked target
        w : array-like, None
            Stacked weight
        """
        return np.stack([x]*self.n, axis=self.axis), y, w


class Flip(Transform):
    def __init__(self, axis, **kwargs):
        """
        Flip data on axis.
        Parameters
        ----------
        axis : int
            ndarray axis to flip
        """
        super().__init__(**kwargs)
        self.axis = axis
    
    def apply(self, x, y=None, w=None):
        """
        Flip data on axis.
        Parameters
        ----------
        x : array-like
            Input
        y : array-like, None
            Target
        w : array-like, None
            Weight
        Returns
        -------
        x : array-like
            Flipped input
        y : array-like, None
            Flipped target
        w : array-like, None
            Flipped weight
        """
        return [None if i is None else np.flip(i, axis=self.axis) for i in (x,y,w)]
    

class Rotate(Transform):
    def __init__(self, degrees, axes=(0,1), order=1, mode="constant", cval=0, **kwargs):
        """
        Rotate 2D or 3D data on axis.
        Parameters
        ----------
        degrees : [float, float]
            mininum to maximum angle of rotation
        axes : int
            Pair of axis to rotate on
            Refer to axes param at https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
        order : int
            The order of interpolation. Between 0-5.
            Refer to order param at https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp
        mode : str, function
            Padding type. Recommend 'reflect', 'constant' or 'nearest'
            Refer to mode param at https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
        cval : __
            Value for constant padding.
            Refer to cval param at https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
        """
        super().__init__(**kwargs)
        self.degrees = degrees
        self.axes    = axes
        self.order   = order
        self.mode    = mode
        self.cval    = cval

    def apply(self, x, y=None, w=None):
        """
        Rotate 2D or 3D data on axis.
        Parameters
        ----------
        x : array-like
            Input
        y : array-like, None
            Target
        w : array-like, None
            Weight
        Returns
        -------
        x : array-like
            Rotated input
        y : array-like, None
            Rotated target
        w : array-like, None
            Rotated weight
        """
        angle = rnd.uniform(*self.degrees)
        return [None if i is None else nd.rotate(input=i,
                                                 angle=angle,
                                                 axes=self.axes,
                                                 reshape=False,
                                                 order=self.order,
                                                 mode=self.mode,
                                                 cval=self.cval) for i in (x,y,w)]


class Round(Transform):
    def __init__(self, decimals, **kwargs):
        """
        Round data.
        Parameters
        ----------
        decimals : int
            Number of decimal to keep
        """
        super().__init__(**kwargs)
        self.decimals = decimals
    
    def apply(self, x, y=None, w=None):
        """
        Round data.
        Parameters
        ----------
        x : array-like
            Input
        y : array-like, None
            Target
        w : array-like, None
            Weight
        Returns
        -------
        x : array-like
            Rounded input
        y : array-like, None
            Rounded target
        w : array-like, None
            Rounded weight
        """
        return [None if i is None else np.round(i, self.decimals) for i in (x,y,w)]


class Tensor(Transform):
    def __init__(self, **kwargs):
        """
        Convert ndarray to tensor.
        """
        super().__init__(**kwargs)
    
    def apply(self, x, y=None, w=None):
        """
        Convert ndarray to tensor.
        Parameters
        ----------
        x : array-like
            Input
        y : array-like, None
            Target
        w : array-like, None
            Weight
        Returns
        -------
        x : array-like
            Input tensor
        y : array-like, None
            Target tensor
        w : array-like, None
            Weight tensor
        """
        return [None if i is None else tf.convert_to_tensor(i) for i in (x,y,w)]


if __name__ == "__main__":
    import tempfile
    import unittest

    class TestTransforms(unittest.TestCase):        
        def test_transform(self):
            abc = (0,0,0)
            with self.assertRaises(NotImplementedError):
                Transform(p=1)(0, 0, 0)
            self.assertEqual((0, 0, 0), Transform(p=0)(0, 0, 0))
        
        def test_compose(self):
            a, b, c = tuple((np.zeros((1,1,1,1)))) * 3
            tf_a, tf_b, tf_c = Compose([Squeeze(axis=0), Squeeze(axis=0), Squeeze(axis=0)])(a, b, c)
            self.assertTrue((a == tf_a).all())
            self.assertTrue((b == tf_b).all())
            self.assertTrue((c == tf_c).all())

        def test_read(self):
            a, b, c = [np.ones((2,2,2))], [np.ones((2,2,2,3))], [np.ones((2,2,2))]
            input_dir = tempfile.TemporaryDirectory()
            input_paths = [input_dir.name + "/{}.tif".format(i) for i in ['a']]

            label_dir = tempfile.TemporaryDirectory()
            label_paths = [label_dir.name + "/{}.tif".format(i) for i in ['b']]

            weight_dir = tempfile.TemporaryDirectory()
            weight_paths = [weight_dir.name + "/{}.tif".format(i) for i in ['c']]

            # inputs and weights
            for path, volume in zip(input_paths + weight_paths, [a, c]):
                tif.imwrite(path, volume, photometric='minisblack')

            # labels
            for path, volume in zip(label_paths, [b]):
                tif.imwrite(path, volume, photometric='rgb')
            
            tf_a, tf_b, tf_c = Read()(input_paths[0], label_paths[0], weight_paths[0])
            self.assertTrue((a == tf_a).all())
            self.assertTrue((b == tf_b).all())
            self.assertTrue((c == tf_c).all())
            
            # cleanup dataset before raising exception
            input_dir.cleanup()
            label_dir.cleanup()
            weight_dir.cleanup()

        def test_pad(self):
            a, b, c = [np.ones((3,4,5))] * 3
            tf_a, tf_b, tf_c = PadResize(shape=[5,5,5])(a, b, c)
            
            self.assertEqual((5, 5, 5), tf_a.shape)
            self.assertEqual((5, 5, 5), tf_b.shape)
            self.assertEqual((5, 5, 5), tf_c.shape)

            self.assertEqual(a.sum(), tf_a.sum())
            self.assertEqual(b.sum(), tf_b.sum())
            self.assertEqual(c.sum(), tf_c.sum())

        def test_sample(self):
            a, b, c = [np.ones((5,5,5))] * 3
            tf_a, tf_b, tf_c = Sample(shape=[3,4,5])(a, b, c)

            self.assertEqual((3, 4, 5), tf_a.shape)
            self.assertEqual((3, 4, 5), tf_b.shape)
            self.assertEqual((3, 4, 5), tf_c.shape)

        def test_normalize(self):
            a, b, c = [np.arange(3)] * 3
            tf_a, tf_b, tf_c = Normalize(-1, 1)(a, b, c)
                       
            self.assertTrue((a - 1 == tf_a).all())
            self.assertTrue((b == tf_b).all())
            self.assertTrue((c == tf_c).all())
        
        def test_standardize(self):
            a, b, c = [np.arange(3)] * 3
            tf_a, tf_b, tf_c = Standardize()(a, b, c)
                       
            self.assertTrue((a != tf_a).all())
            self.assertTrue((b == tf_b).all())
            self.assertTrue((c == tf_c).all())
        
        def test_squeeze(self):
            a, b, c = [np.ones((1,3,3))] * 3
            tf_a, tf_b, tf_c = Squeeze(axis=0)(a, b, c)
                       
            self.assertTrue((a[0] == tf_a).all())
            self.assertTrue((b[0] == tf_b).all())
            self.assertTrue((c[0] == tf_c).all())
        
        def test_stack(self):
            a, b, c = [np.ones((1,3,3))] * 3
            tf_a, tf_b, tf_c = Stack(axis=-1, n=1)(a, b, c)
                       
            self.assertTrue((np.expand_dims(a, axis=-1) == tf_a).all())
            self.assertTrue((b == tf_b).all())
            self.assertTrue((c == tf_c).all())
        
        def test_flip(self):
            array = np.zeros((2,2,2))
            array[0,0,0] = 1
            a, b, c = [array.copy()] * 3
            tf_a, tf_b, tf_c = Compose([Flip(axis=0), Flip(axis=1), Flip(axis=2)])(a, b, c)
            tf_array = np.zeros((2,2,2))
            tf_array[-1,-1,-1] = 1

            self.assertTrue((tf_array == tf_a).all())
            self.assertTrue((tf_array == tf_b).all())
            self.assertTrue((tf_array == tf_c).all())
        
        def test_rotate(self):
            array = np.zeros((2,2,2))
            array[0,0,0] = 1
            a, b, c = [array.copy()] * 3
            tf_a, tf_b, tf_c = Compose([Rotate(degrees=[90,90], axes=(1,2), order=3), Round(decimals=5)])(a, b, c)
            tf_array = np.zeros((2,2,2))
            tf_array[0,1,0] = 1

            self.assertTrue((tf_array == tf_a).all())
            self.assertTrue((tf_array == tf_b).all())
            self.assertTrue((tf_array == tf_c).all())

        def test_tensor(self):
            a, b, c = [np.ones((3,4,5))] * 3
            tf_a, tf_b, tf_c = Tensor()(a, b, c)
            
            self.assertTrue((a.shape == tf_a.shape))
            self.assertTrue((b.shape == tf_b.shape))
            self.assertTrue((c.shape == tf_c.shape))

            self.assertIsInstance(tf_a, tf.Tensor)
            self.assertIsInstance(tf_b, tf.Tensor)
            self.assertIsInstance(tf_c, tf.Tensor)
            
    unittest.main()
