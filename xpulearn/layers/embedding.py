import numpy

from . import Layer
from .. import xp, xpx, Parameter


class Embedding(Layer):

    """Embedding layer

    Attributes:
        vocab_size (int): vocabulary size
        num_units (int): dimension of embedding vectors
        vocab_ids (2D array): shape [batch size, timesteps]
    NN Parameters:
        W (2D array): embedding matrix, shape [vocabulary size, embedding size]
    """

    def __init__(self, vocab_size, num_units):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.vocab_ids = None
        self.params = {}
        self.grads = {}

    def init_params(self, layer_id, input_dim, dtype):
        # TODO: input_dim is not used
        if 'W' not in self.params and 'W' not in self.grads:
            w_id = '{}_W'.format(layer_id)
            weight_shape = (self.vocab_size, self.num_units)
            self.params['W'] = Parameter(
                w_id, xp.random.randn(*weight_shape).astype(dtype) / 10)
            self.grads['W'] = Parameter(
                w_id, xp.zeros(weight_shape, dtype=dtype))

    def forwardprop(self, X):
        """
        Arguments:
            X (2D array): shape [batch_size, timesteps]
        Returns:
            X_out (3D array): shape [batch size, timesteps, embedding size]
        """
        batch_size = X.shape[0]
        timesteps = X.shape[1]
        X_out = xp.empty((batch_size, timesteps, self.num_units))
        self.vocab_ids = xp.zeros((batch_size, timesteps), dtype=xp.int)
        for i in range(batch_size):
            self.vocab_ids[i][...] = X[i]
            X_out[i][...] = self.params['W'].data[self.vocab_ids[i]]
        return X_out

    def backprop(self, dX):
        """
        Arguments:
            dX (3D array): shape [batch size, timesteps, embedding size]
        """
        batch_size = dX.shape[0]
        self.grads['W'].data[...] = 0
        for i in range(batch_size):
            if xp == numpy:
                xp.add.at(self.grads['W'].data, self.vocab_ids[i], dX[i])
            else:
                xpx.scatter_add(self.grads['W'].data, self.vocab_ids[i], dX[i])
