import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):
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
