from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into disjoint random training and test subsets.

    The training set contains ``int(len(x) * tr_fraction)`` samples; the
    remaining samples are assigned to the test set.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 2:
        raise ValueError("x must be a 2-dimensional array")
    if y.ndim != 1:
        raise ValueError("y must be a 1-dimensional array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples")
    if (
        isinstance(tr_fraction, (bool, np.bool_))
        or not isinstance(tr_fraction, (int, float, np.integer, np.floating))
        or not np.isfinite(tr_fraction)
        or not 0 <= tr_fraction <= 1
    ):
        raise ValueError("tr_fraction must be a number between 0 and 1")

    indices = np.random.permutation(x.shape[0])
    n_train = int(x.shape[0] * tr_fraction)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]
