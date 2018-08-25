import time

import numpy as np

from . import optimizers
from .layers import Activation, Loss


class Model(object):

    def __init__(self, loss, input_shape, optimizer=optimizers.SGD()):
        self.loss_layer = Loss(loss)
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def init_params(self):
        # TODO: need initializer
        prev_layer_shape = self.input_shape
        for layer_id, layer in enumerate(self.layers):
            # activation layer has no params
            if isinstance(layer, Activation):
                continue
            layer.init_params(layer_id, prev_layer_shape)
            self.optimizer.init_params(layer.params)
            # TODO: How about other layers ?
            prev_layer_shape = layer.params['W'].data.shape[:-1]

    def predict(self, X):
        for layer in self.layers:
            X = layer.forwardprop(X)
        return X

    def loss(self, X, Y):
        Y_pred = self.predict(X)
        return self.loss_layer.forwardprop(Y_pred, Y)

    def fit(self, X, Y, epochs=10, batch_size=16, verbose=False):
        self.init_params()
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
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
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
        for layer in self.layers:
            X = layer.forwardprop(X)
        # backprop
        dout = self.loss_layer.backprop(X, Y)
        for layer in reversed(self.layers):
            dout = layer.backprop(dout)
            # activation layer has no params
            if isinstance(layer, Activation):
                continue
            self.optimizer.update_params(layer.params, layer.grads)
