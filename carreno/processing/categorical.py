import numpy as np

def categorical_to_sparse(x):
    """Create a multi-class ndarray (exclusive class, aka sparse categorical)
    from given multi-label ndarray (non-exclusive class, aka categorical)
    Parameters
    ----------
    x : ndarray
        categorical ndarray
    Returns
    -------
    y : ndarray
        sparse categorical ndarray
    """
    y = np.argmax(x, axis=-1) + 1

    return y


def sparse_to_categorical(x, n_class=None):
    """Create a multi-label ndarray (non-exclusive class, aka categorical)
    from given multi-class ndarray (exclusive class, aka sparse categorical)
    Parameters
    ----------
    x : ndarray
        multi-class ndarray
    n_class : int
        number of classes, default to maximum value in ndarray
    Returns
    -------
    y : ndarray
        multi-label ndarray
    """
    if n_class is None:
        n_class = x.max()

    y = np.zeros((*x.shape, n_class), dtype=float)

    for i in range(n_class):
        lb = i + 1
        y[x == lb, i] = 1

    return y


def categorical_multiclass(x):
    """
    Keep ndarray categorical format and makes classes exclusive (soft to hard classification).
    (multi-label to multi-class).
    Parameters
    ----------
    x : ndarray
        ndarray to classify
    Returns
    -------
    y : ndarray
        boolean classified ndarray
    """
    n_class = x.shape[-1]

    # from multi-label to multi-class
    y = categorical_to_sparse(x)

    # keep categorical format
    y = sparse_to_categorical(y, n_class).astype(bool)
    
    return y


if __name__ == "__main__":
    import unittest

    class TestCategorical(unittest.TestCase):
        def test_categorical2sparse(self):
            x = np.array([[1,0],[0,1],[0.5,0.5],[0.25,0.75]])
            y = np.array([1,2,1,2])
            self.assertTrue((categorical_to_sparse(x) == y).all())
        
        def test_sparse2categorical(self):
            x = np.array([1,2,1,2])
            y = np.array([[1,0],[0,1],[1,0],[0,1]])
            self.assertTrue((sparse_to_categorical(x) == y).all())
        
        def test_categorical_multiclass(self):
            x = np.array([[1,0],[0,1],[0.5,0.5],[0.25,0.75]])
            y = np.array([[1,0],[0,1],[1,0],[0,1]])
            self.assertTrue((categorical_multiclass(x) == y).all())
            
    unittest.main()