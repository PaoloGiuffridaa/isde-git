import numpy as np
from os import path
import unittest

from fun_utils import load_data, split_data
from classifiers import NMC


class TestNMC(unittest.TestCase):

    def setUp(self):
        # Loading MNIST dataset
        mnist_path = path.join(
            path.dirname(__file__), '..', '../data', 'mnist_data.csv')
        x, y = load_data(filename=mnist_path)
        # Rescale data in 0-1
        self.x = x / 255
        self.y = y
        self.clf = NMC()

    def test_split_data(self):
        xtr, ytr, xts, yts = split_data(self.x, self.y, tr_fraction=0.5)

        assert type(xtr) == np.ndarray
        assert type(ytr) == np.ndarray
        assert type(xts) == np.ndarray
        assert type(yts) == np.ndarray

        assert xtr.ndim == 2
        assert ytr.ndim == 1
        assert xts.ndim == 2
        assert yts.ndim == 1

        tr_size = int(self.x.shape[0] * 0.5)
        ts_size = self.x.shape[0] - xtr.shape[0]

        assert xtr.shape[0] == tr_size
        assert ytr.size == tr_size
        assert xts.shape[0] == ts_size
        assert yts.size == ts_size

        assert xtr.shape[1] == self.x.shape[1]
        assert xts.shape[1] == self.x.shape[1]

    def test_split_data_returns_disjoint_aligned_subsets(self):
        x = np.arange(10).reshape(5, 2)
        y = np.arange(5)

        xtr, ytr, xts, yts = split_data(x, y, tr_fraction=0.5)

        train_indices = {tuple(row) for row in xtr}
        test_indices = {tuple(row) for row in xts}
        all_indices = {tuple(row) for row in x}
        assert xtr.shape[0] == 2
        assert xts.shape[0] == 3
        assert train_indices.isdisjoint(test_indices)
        assert train_indices | test_indices == all_indices
        assert all(np.array_equal(row, x[label]) for row, label in zip(xtr, ytr))
        assert all(np.array_equal(row, x[label]) for row, label in zip(xts, yts))

    def test_fit(self):
        self.clf.fit(self.x, self.y)
        assert self.clf.centroids is not None
        assert self.clf.centroids.shape[0] == np.unique(self.y).size
        assert self.clf.centroids.shape[1] == self.x.shape[1]

    def test_fit_computes_centroids_for_noncontiguous_labels(self):
        x = np.array([[0.0, 2.0], [2.0, 4.0], [8.0, 10.0]])
        y = np.array([2, 2, 7])

        self.clf.fit(x, y)

        np.testing.assert_array_equal(self.clf.class_labels, [2, 7])
        np.testing.assert_allclose(self.clf.centroids, [[1.0, 3.0], [8.0, 10.0]])

    def test_predict(self):
        self.clf.fit(self.x, self.y)
        yc = self.clf.predict(self.x)
        assert yc.ndim == 1
        assert yc.shape[0] == self.x.shape[0]

    def test_predict_returns_nearest_class_labels(self):
        x = np.array([[0.0], [2.0], [10.0], [12.0]])
        y = np.array([1, 1, 5, 5])
        self.clf.fit(x, y)

        predictions = self.clf.predict(np.array([[1.0], [11.0], [3.0]]))

        np.testing.assert_array_equal(predictions, [1, 5, 1])
