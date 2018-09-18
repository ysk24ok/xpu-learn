from . import Activation, Dense
from .base import Layer

from .. import xp


class RNNCell(object):

    def __init__(self, layers):
        self.layers = layers

    def forwardprop(self, X, H):
        """
        Arguments:
            X (2D array): shape [batch size, #units of input-side layer]
            H (2D array): shape [batch size, #units of this layer]
        Returns:
            X_out (2D array): shape [batch size, #units of this layer]
            H_out (2D array): shape [batch size, #units of this layer]
        """
        H_out = self.layers['Wh'].forwardprop(H)
        X_out = self.layers['Wx'].forwardprop(X)
        H_out = self.layers['A'].forwardprop(H_out + X_out)
        X_out = H_out
        return X_out, H_out

    def backprop(self, dX, dH):
        """
        Arguments:
            dX (2D array): shape [batch size, #units of this layer]
            dH (2D array): shape [batch size, #units of this layer]
        Returns:
            dX_out (2D array): shape [batch size, #units of input-side layer]
            dH_out (2D array): shape [batch size, #units of this layer]
        """
        dX = self.layers['A'].backprop(dX + dH)
        dH_out = self.layers['Wh'].backprop(dX)
        dX_out = self.layers['Wx'].backprop(dX)
        return dX_out, dH_out


class RNN(Layer):

    """Recurrent Layer

    Attributes:
        num_units (int): #units of this layer
        input_dim (int): #units of input-side layer
        timesteps (int): timesteps
        return_sequences (bool): Return 3D array in forward propgation if True,
                                 return 2D array otherwise
    NN Parameters:
        Wh (2D array): shape [#units of this layer, #units of this layer])
        bh (2D array): shape [#units of this layer, 1]
        Wx (2D array): shape [#units of this layer, #units of input-side layer]
        bx (2D array): shape [#units of this layer, 1]
    """

    def __init__(
            self, num_units,
            activation='tanh',
            recurrent_weight_initializer='he',
            recurrent_bias_initializer='zeros',
            weight_initializer='he',
            bias_initializer='zeros',
            return_sequences=False):
        super(RNN, self).__init__()
        self.num_units = num_units
        self.input_dim = None
        self.timesteps = None
        self.return_sequences = return_sequences
        self.cells = []
        self.layers = {
            'Wh': Dense(self.num_units,
                        weight_initializer=recurrent_weight_initializer,
                        bias_initializer=recurrent_bias_initializer),
            'Wx': Dense(self.num_units,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer),
            'A': Activation(activation)
        }
        self.params = {}
        self.grads = {}

    def init_params(self, layer_id, input_dim, dtype):
        self.dtype = dtype
        self.input_dim = input_dim
        # Wh
        self.layers['Wh'].init_params(
            '{}_Wh'.format(layer_id), self.num_units, dtype)
        self.params['Wh'] = self.layers['Wh'].params['W']
        self.params['bh'] = self.layers['Wh'].params['b']
        self.grads['Wh'] = self.layers['Wh'].grads['W']
        self.grads['bh'] = self.layers['Wh'].grads['b']
        # Wx
        self.layers['Wx'].init_params(
            '{}_Wx'.format(layer_id), input_dim, dtype)
        self.params['Wx'] = self.layers['Wx'].params['W']
        self.params['bx'] = self.layers['Wx'].params['b']
        self.grads['Wx'] = self.layers['Wx'].grads['W']
        self.grads['bx'] = self.layers['Wx'].grads['b']

    def forwardprop(self, X):
        """
        Arguments:
            X (3D array):
                shape [batch size, timesteps, #units of input-side layer]
        Returns:
            (2D array or 3D array):
                3D array of shape [batch size, timesteps, #units of this layer]
                if return_sequences is True,
                2D array of shape [batch size, #units of this layer] otherwise
        """
        batch_size = X.shape[0]
        self.timesteps = X.shape[1]
        H = xp.zeros((batch_size, self.num_units), dtype=self.dtype)
        X_out = xp.empty(
            (batch_size, self.timesteps, self.num_units), dtype=self.dtype)
        self.cells = []
        for t in range(self.timesteps):
            cell = RNNCell(self.layers)
            self.cells.append(cell)
            X_out[:, t, :], H = cell.forwardprop(X[:, t, :], H)
        if self.return_sequences is True:
            return X_out
        return X_out[:, -1, :]

    def backprop(self, dX):
        """
        Arguments:
            dX (2D array or 3D array):
                3D array of shape [batch size, timesteps, #units of this layer]
                if return_sequences is True,
                2D array of shape [batch size, #units of this layer] otherwise
        Returns:
            dX_out (3D array):
                shape [batch size, timesteps, #units of input-side layer]
        """
        batch_size = dX.shape[0]
        if self.return_sequences is False:
            dX_3D = xp.zeros(
                (batch_size, self.timesteps, self.num_units), dtype=self.dtype)
            dX_3D[:, -1, :][...] = dX
            dX = dX_3D
        dH = xp.zeros((batch_size, self.num_units), dtype=self.dtype)
        dX_out = xp.empty(
            (batch_size, self.timesteps, self.input_dim), dtype=self.dtype)
        # initialize grads
        for grad in self.grads.values():
            grad.data[...] = 0
        for t in reversed(range(self.timesteps)):
            cell = self.cells[t]
            dX_out[:, t, :][...], dH[...] = cell.backprop(dX[:, t, :], dH)
            # Wh
            self.grads['Wh'].data += self.layers['Wh'].grads['W'].data
            self.grads['bh'].data += self.layers['Wh'].grads['b'].data
            # Wx
            self.grads['Wx'].data += self.layers['Wx'].grads['W'].data
            self.grads['bx'].data += self.layers['Wx'].grads['b'].data
        return dX_out
