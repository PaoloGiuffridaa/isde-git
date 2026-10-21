import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Nearest Mean Centroid (NMC) classifier.

    The classifier estimates one feature-wise mean centroid per class from
    training data. Each prediction is the label of the closest centroid in
    Euclidean distance.

    The read-only :attr:`centroids` and :attr:`class_labels` properties expose
    the learned model state after fitting.
    """

    def __init__(self):
        """Initialize an unfitted NMC classifier."""
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        """Return the learned class centroids, or ``None`` if not fitted."""
        return self._centroids

    @property
    def class_labels(self):
        """Return labels corresponding to centroid rows, or ``None`` if unfitted."""
        return self._class_labels

    def fit(self, xtr, ytr):
        """
        Compute a mean feature vector for each class in the training set.

        Parameters
        ----------
        xtr : array-like of shape (n_samples, n_features)
            Training feature matrix.
        ytr : array-like of shape (n_samples,)
            Class label for each training sample.

        Returns
        -------
        self : NMC
            The fitted classifier.

        Raises
        ------
        ValueError
            If the feature matrix or labels have invalid dimensions, their
            sample counts differ, or the training set is empty.
        """
        xtr = np.asarray(xtr)
        ytr = np.asarray(ytr)

        if xtr.ndim != 2:
            raise ValueError("xtr must be a 2-dimensional array")
        if ytr.ndim != 1:
            raise ValueError("ytr must be a 1-dimensional array")
        if xtr.shape[0] != ytr.shape[0]:
            raise ValueError("xtr and ytr must contain the same number of samples")
        if xtr.shape[0] == 0:
            raise ValueError("training data must contain at least one sample")

        class_labels = np.unique(ytr)
        centroids = np.stack(
            [xtr[ytr == label].mean(axis=0) for label in class_labels]
        )
        self._class_labels = class_labels
        self._centroids = centroids
        return self

    def predict(self, xts):
        """
        Predict labels using the nearest learned class centroid.

        Parameters
        ----------
        xts : array-like of shape (n_samples, n_features)
            Feature matrix to classify. Its feature count must match the
            training data used by :meth:`fit`.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted class label for each input sample.

        Raises
        ------
        RuntimeError
            If the classifier has not been fitted.
        ValueError
            If ``xts`` is not two-dimensional or has a different number of
            features from the training data.
        """
        if self._centroids is None or self._class_labels is None:
            raise RuntimeError("fit must be called before predict")

        xts = np.asarray(xts)
        if xts.ndim != 2:
            raise ValueError("xts must be a 2-dimensional array")
        if xts.shape[1] != self._centroids.shape[1]:
            raise ValueError(
                "xts must have the same number of features as training data"
            )

        distances = euclidean_distances(xts, self._centroids)
        return self._class_labels[np.argmin(distances, axis=1)]
