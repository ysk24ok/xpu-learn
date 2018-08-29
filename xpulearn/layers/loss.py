import numpy as np

from .activation import SigmoidActivation, SoftmaxActivation
from .base import Layer
from ..functions import clip_before_log


class Loss(Layer):

    def __init__(self, loss, pre_activation=None):
        super(Loss, self).__init__()
        if loss == 'mse':
            self.loss = MeanSqueredError()
        elif loss == 'binary_crossentropy':
            self.loss = self._binary_crossentropy(pre_activation)
        elif loss == 'categorical_crossentropy':
            self.loss = self._categorical_crossentropy(pre_activation)
        else:
            raise ValueError('Invalid value for loss: {}'.format(loss))

    def _binary_crossentropy(self, pre_activation):
        if pre_activation is None:
            return BinaryCrossEntropy()
        if isinstance(pre_activation.activation, SigmoidActivation):
            return BinaryCrossEntropyWithSigmoid(pre_activation)
        return BinaryCrossEntropy()

    def _categorical_crossentropy(self, pre_activation):
        if pre_activation is None:
            return CategoricalCrossEntropy()
        if isinstance(pre_activation.activation, SoftmaxActivation):
            return CategoricalCrossEntropyWithSoftmax(pre_activation)
        return CategoricalCrossEntropy()

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
        X_clipped = clip_before_log(X.flatten(), self.dtype)
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


class BinaryCrossEntropyWithSigmoid(LossLayer):

    def __init__(self, pre_activation):
        self.pre_activation = pre_activation
        self.pre_activation.skip = True
        self.loss = BinaryCrossEntropy()

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
        X = self.pre_activation.forwardprop(X)
        return self.loss.forwardprop(X, y)

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
        return (self.pre_activation.activation.X_out - Y) / batch_size


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
        X_clipped = clip_before_log(X, self.dtype)
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


class CategoricalCrossEntropyWithSoftmax(LossLayer):

    def __init__(self, pre_activation):
        self.pre_activation = pre_activation
        self.pre_activation.skip = True
        self.loss = CategoricalCrossEntropy()

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
        X = self.pre_activation.forwardprop(X)
        return self.loss.forwardprop(X, Y)

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
        return (self.pre_activation.activation.X_out - Y) / batch_size
