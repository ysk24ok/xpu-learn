import time

from . import xp
from .layers import Activation, Dropout, Loss


class Model(object):

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss, dtype='float32'):
        if dtype not in ('float64', 'float32'):
            raise ValueError("dtype must be either 'float64' or 'float32'")
        self.optimizer = optimizer
        self.loss_layer = Loss(loss, self.layers[-1])
        self.init_params(dtype)
        self.set_dtype_to_layers(dtype)

    def init_params(self, dtype):
        # TODO: need initializer
        prev_layer_shape = self.input_shape
        for layer_id, layer in enumerate(self.layers):
            # activation and dropout layer has no params
            if isinstance(layer, Activation) or isinstance(layer, Dropout):
                continue
            layer.init_params(layer_id, prev_layer_shape, dtype)
            self.optimizer.init_params(layer.params, dtype)
            # TODO: How about other layers ?
            prev_layer_shape = layer.params['W'].data.shape[:-1]

    def set_dtype_to_layers(self, dtype):
        for layer in self.layers:
            if isinstance(layer, Activation):
                layer.activation.dtype = dtype
            else:
                layer.dtype = dtype
        self.loss_layer.loss.dtype = dtype

    def predict(self, X, training=False):
        for layer in self.layers:
            if layer.skip is True:
                continue
            if isinstance(layer, Dropout):
                X = layer.forwardprop(X, training=training)
                continue
            X = layer.forwardprop(X)
        return X

    def loss(self, X, Y):
        Y_pred = self.predict(X)
        return self.loss_layer.forwardprop(Y_pred, Y)

    def fit(self, X, Y, epochs=10, batch_size=16, verbose=False):
        num_samples = X.shape[0]
        num_epochs = 0
        elapsed_per_epoch = 0
        while num_epochs < epochs:
            if verbose is True:
                loss = self.loss(X, Y)
                if num_epochs != 0:
                    print(
                        'epoch: {0:>3}, loss: {1:.4f}, '
                        'elapsed: {2:.2f} sec'.format(
                            num_epochs, loss, elapsed_per_epoch))
                else:
                    print('epoch: {0:>3}, loss: {1:.4f}'.format(
                        num_epochs, loss))
            stime = time.time()
            indices = xp.arange(num_samples)
            xp.random.shuffle(indices)
            num_iters = 0
            while True:
                p_idx = batch_size * num_iters
                n_idx = p_idx + batch_size
                idx = indices[p_idx:n_idx]
                X_minibatch = X[idx]
                Y_minibatch = Y[idx]
                if X_minibatch.shape[0] < batch_size:
                    break
                self.fit_on_minibatch(X_minibatch, Y_minibatch)
                num_iters += 1
            num_epochs += 1
            etime = time.time()
            elapsed_per_epoch = etime - stime

    def fit_on_minibatch(self, X, Y):
        # forwardprop
        X = self.predict(X, training=True)
        if self.layers[-1].skip is True:
            self.layers[-1].forwardprop(X)
        # backprop
        dout = self.loss_layer.backprop(X, Y)
        for layer in reversed(self.layers):
            if layer.skip is True:
                continue
            dout = layer.backprop(dout)
            # activation and dropout layer has no params
            if isinstance(layer, Activation) or isinstance(layer, Dropout):
                continue
            self.optimizer.update_params(layer.params, layer.grads)
