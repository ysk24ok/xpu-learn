from .base import Layer
from . import Activation
from .. import xp, Parameter
from ..initializers import Initializer


class Dense(Layer):

    """Fully connected layer

    NN Parameters:
        W (2D array): shape [#units of this layer, #units of input-side layer]
        b (2D array): shape [#units of this layer, 1]
    """

    def __init__(
            self, num_units, activation='linear',
            weight_initializer='he', bias_initializer='zeros'):
        """
        Arguments:
            num_units (int): #units of this layer
            activation (str): activation
            weight_initializer: initializer type for W
            bias_initializer: initializer type for b
        """
        super(Dense, self).__init__()
        self.num_units = num_units
        self.activation = Activation(activation)
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.cache = {'X': None}
        self.params = {}
        self.grads = {}

    def init_params(self, layer_id, input_dim, dtype):
        weight_initializer = Initializer(self.weight_initializer, dtype)
        bias_initializer = Initializer(self.bias_initializer, dtype)
        zeros_initializer = Initializer('zeros', dtype)
        # weight
        if 'W' not in self.params and 'W' not in self.grads:
            w_id = '{}_W'.format(layer_id)
            weight_shape = (self.num_units, input_dim)
            self.params['W'] = Parameter(
                w_id, weight_initializer(weight_shape))
            self.grads['W'] = Parameter(w_id, zeros_initializer(weight_shape))
        # bias
        if 'b' not in self.params and 'b' not in self.grads:
            b_id = '{}_b'.format(layer_id)
            bias_shape = (self.num_units, 1)
            self.params['b'] = Parameter(b_id, bias_initializer(bias_shape))
            self.grads['b'] = Parameter(b_id, zeros_initializer(bias_shape))

    def forwardprop(self, X):
        """
        Arguments:
            X (2D array): shape [batch size, #units of input-side layer]
        Returns:
            (2D array): shape [batch_size, #units of this layer]
        """
        self.cache['X'] = X
        W = self.params['W'].data
        b = self.params['b'].data
        X_out = X @ W.T
        X_out += b.T
        return self.activation.forwardprop(X_out)

    def backprop(self, dX):
        """
        Arguments:
            dX (2D array): shape [batch size, #units of this layer]
        Returns:
            (2D array): shape [batch size, #units of input-side layer]
        """
        batch_size = dX.shape[0]
        dX = self.activation.backprop(dX)
        self.grads['W'].data[...] = dX.T @ self.cache['X']
        self.grads['W'].data /= batch_size
        self.grads['b'].data[...] = xp.sum(dX.T, axis=1, keepdims=True)
        self.grads['b'].data /= batch_size
        return dX @ self.params['W'].data
