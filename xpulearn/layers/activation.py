from .base import Layer
from .. import xp
from ..functions import clip_before_exp


class Activation(Layer):

    def __init__(self, activation):
        super(Activation, self).__init__()
        if activation == 'linear':
            self.activation = LinearActivation()
        elif activation == 'sigmoid':
            self.activation = SigmoidActivation()
        elif activation == 'softmax':
            self.activation = SoftmaxActivation()
        elif activation == 'tanh':
            self.activation = TanhActivation()
        elif activation == 'relu':
            self.activation = ReLUActivation()
        else:
            msg = 'Invalid value for activation: {}'.format(activation)
            raise ValueError(msg)

    def forwardprop(self, X_in):
        """
        Arguments:
            X_in (numpy.ndarray or cupy.core.core.ndarray):
                2D array of shape [batch size, #units of input-side layer]
        Returns:
            (numpy.ndarray or cupy.core.core.ndarray):
                2D array of the same shape as `X_in`
        """
        self.X_in = X_in
        return self.activation.forwardprop(X_in)

    def backprop(self, dout):
        """
        Arguments:
            dout (numpy.ndarray or cupy.core.core.ndarray):
                2D array of shape [batch size, #units of output-side layer]
        Returns:
            (numpy.ndarray or cupy.core.core.ndarray):
                2D array of the same shape as `dout`
        """
        if isinstance(self.activation, LinearActivation):
            return dout
        return dout * self.activation.backprop(self.X_in)


class BaseActivation(Layer):

    pass


class LinearActivation(BaseActivation):

    def forwardprop(self, X_in):
        return X_in

    def backprop(self, X_in):
        return xp.ones_like(X_in)


class SigmoidActivation(BaseActivation):

    def forwardprop(self, X_in):
        X_in = clip_before_exp(-X_in, self.dtype)
        self.X_out = 1 + xp.exp(X_in)
        self.X_out **= -1
        return self.X_out

    def backprop(self, X_in):
        X_out = self.X_out
        if X_out is None:
            X_out = self.forwardprop(X_in)
        return X_out * (1 - X_out)


class SoftmaxActivation(BaseActivation):

    def forwardprop(self, X_in):
        # Avoid overflow encountered in exp
        self.X_out = X_in - X_in.max(axis=1, keepdims=True)
        self.X_out[...] = xp.exp(self.X_out)
        self.X_out /= xp.sum(self.X_out, axis=1, keepdims=True)
        return self.X_out

    def backprop(self, X_in):
        X_out = self.X_out
        if X_out is None:
            X_out = self.forwardprop(X_in)
        return X_out * (1 - X_out)


class TanhActivation(BaseActivation):

    def forwardprop(self, X_in):
        self.X_out = xp.tanh(X_in)
        return self.X_out

    def backprop(self, X_in):
        X_out = self.X_out
        if X_out is None:
            X_out = self.forwardprop(X_in)
        return 1 - xp.square(X_out)


class ReLUActivation(BaseActivation):

    def forwardprop(self, X_in):
        self.X_out = xp.maximum(X_in, 0)
        return self.X_out

    def backprop(self, X_in):
        dX = xp.zeros_like(X_in)
        dX[self.X_out > 0.0] = 1.0
        return dX
