import numpy as np

def categorical_multiclass(x):
    """Keep ndarray categorical format and makes classes exclusive (soft to hard classification)
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
    