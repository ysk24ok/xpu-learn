import numpy as np

from .base import Layer


_eps = np.finfo(float).eps


class Loss(Layer):

    def __init__(self, loss):
        if loss == 'mse':
            self.loss = MeanSqueredError()
        elif loss == 'binary_crossentropy':
            self.loss = BinaryCrossEntropy()
        elif loss == 'categorical_crossentropy':
            self.loss = CategoricalCrossEntropy()
        else:
            raise ValueError('Invalid value for loss: {}'.format(loss))

    def forwardprop(self, X, Y):
        return self.loss.forwardprop(X, Y)

    def backprop(self, X, Y):
        return self.loss.backprop(X, Y)


class LossLayer(Layer):

    pass


class MeanSqueredError(LossLayer):

    def forwardprop(self, X, y):
        """
        Arguments:
            X (numpy.ndarray):
                2D array of shape [batch size, 1]
            y (numpy.ndarray):
                1D array of shape [batch size,]
        Returns:
            (numpy.float64): loss
        """
        batch_size = X.shape[0]
        err = X.flatten() - y
        return np.sum(err ** 2) / (2 * batch_size)

    def backprop(self, X, y):
        """
        Arguments:
            X (numpy.ndarray):
                2D array of shape [batch size, 1]
            y (numpy.ndarray):
                1D array of shape [batch size,]
        Returns:
            (2d array, [batch size, 1])
        """
        batch_size = X.shape[0]
        return (X - y.reshape(batch_size, 1)) / batch_size


class BinaryCrossEntropy(LossLayer):

    def forwardprop(self, X, y):
        """
        Arguments:
            X (numpy.ndarray):
                2D array of shape [batch size, 1]
            y (numpy.ndarray):
                1D array of shape [batch size,]
        Returns:
            (numpy.float64): loss
        """
        batch_size = X.shape[0]
        X_clipped = np.clip(X.flatten(), _eps, 1-_eps)
        loss_pos = y * np.log(X_clipped)
        loss_neg = (1-y) * np.log(1-X_clipped)
        return -np.sum(loss_pos + loss_neg) / batch_size

    def backprop(self, X, y):
        """
        Arguments:
            X (numpy.ndarray):
                2D array of shape [batch size, 1]
            y (numpy.ndarray):
                1D array of shape [batch size,]
        Returns:
            (numpy.ndarray): 2D array of shape [batch size, 1]
        """
        batch_size = X.shape[0]
        Y = y.reshape(batch_size, 1)
        dout = -Y / X
        dout += (1-Y) / (1-X)
        dout /= batch_size
        return dout


class CategoricalCrossEntropy(LossLayer):

    def forwardprop(self, X, Y):
        """
        Arguments:
            X (numpy.ndarray):
                2D array of shape [batch size, number of labels])
            Y (numpy.ndarray):
                2D array of shape [batch size, number of labels])
        Returns:
            (numpy.float64): loss
        """
        batch_size = X.shape[0]
        X_clipped = np.clip(X, _eps, 1-_eps)
        loss_pos = Y * np.log(X_clipped)
        loss_neg = (1-Y) * np.log(1-X_clipped)
        return -np.sum(loss_pos + loss_neg) / batch_size

    def backprop(self, X, Y):
        """
        Arguments:
            X (numpy.ndarray):
                2D array of shape [batch size, number of labels])
            Y (numpy.ndarray):
                2D array of shape [batch size, number of labels])
        Returns:
            (numpy.ndarray):
                2D array of shape [batch size, number of labels])
        """
        batch_size = X.shape[0]
        dout = -Y / X
        dout += (1-Y) / (1-X)
        dout /= batch_size
        return dout
