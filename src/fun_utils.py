from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load a labeled dataset from a CSV file.

    The first CSV column is treated as the class label, and all remaining
    columns are treated as features.

    Parameters
    ----------
    filename : str or path-like
        Path to the CSV file to load.

    Returns
    -------
    X : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : numpy.ndarray of shape (n_samples,)
        Class label for each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into disjoint random training and test subsets.

    A single random permutation of sample indices is used to split features
    and labels together, preserving their correspondence. The training set
    contains ``int(len(x) * tr_fraction)`` samples; the remaining samples are
    assigned to the test set.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Class labels corresponding to rows of ``x``.
    tr_fraction : float, default=0.5
        Fraction of samples assigned to the training set. Must be between
        zero and one, inclusive.

    Returns
    -------
    x_train : numpy.ndarray of shape (n_train, n_features)
        Training features.
    y_train : numpy.ndarray of shape (n_train,)
        Training labels.
    x_test : numpy.ndarray of shape (n_samples - n_train, n_features)
        Test features.
    y_test : numpy.ndarray of shape (n_samples - n_train,)
        Test labels.

    Raises
    ------
    ValueError
        If the inputs have incompatible dimensions or lengths, or if
        ``tr_fraction`` is not a number between zero and one.
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
