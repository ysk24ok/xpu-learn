from . import Parameter

from abc import abstractmethod

import numpy as np


class Optimizer(object):

    @abstractmethod
    def init_param(self, param, dtype):
        """
        Arguments:
            param (xpulearn.Parameter)
        """
        pass

    def init_params(self, params, dtype):
        """
        Arguments:
            params (dict[str, xpulearn.Parameter])
        """
        for param in params.values():
            self.init_param(param, dtype)

    @abstractmethod
    def update(self, param, grad):
        """
        Arguments:
            param (xpulearn.Parameter)
            grad (xpulearn.Parameter)
        """
        pass

    def update_params(self, params, grads):
        """
        Arguments:
            params (dict[str, xpulearn.Parameter])
            grads (dict[str, xpulearn.Parameter])
        """
        for k, param in params.items():
            grad = grads[k]
            self.update(param, grad)


class SGD(Optimizer):

    def __init__(self, lr=0.01):
        self.lr = lr

    def init_param(self, param, dtype):
        pass

    def update(self, param, grad):
        param.data -= self.lr * grad.data


class MomentumSGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def init_param(self, param, dtype):
        if param.id not in self.v:
            self.v[param.id] = Parameter(
                param.id, np.zeros_like(param.data, dtype=dtype))

    def update(self, param, grad):
        self.v[param.id].data *= self.momentum
        self.v[param.id].data += (1 - self.momentum) * grad.data
        param.data -= self.lr * self.v[param.id].data


class RMSprop(Optimizer):

    def __init__(self, lr=0.01, alpha=0.99, eps=1e-08):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.s = {}

    def init_param(self, param, dtype):
        if param.id not in self.s:
            self.s[param.id] = Parameter(
                param.id, np.zeros_like(param.data, dtype=dtype))

    def update(self, param, grad):
        self.s[param.id].data *= self.alpha
        self.s[param.id].data += (1 - self.alpha) * np.square(grad.data)
        grad.data *= (np.sqrt(self.s[param.id].data) + self.eps) ** -1
        param.data -= self.lr * grad.data


class AdaGrad(Optimizer):

    def __init__(self, lr=0.001, eps=1e-08):
        self.lr = lr
        self.eps = eps
        self.v = {}

    def init_param(self, param, dtype):
        if param.id not in self.v:
            self.v[param.id] = Parameter(
                param.id, np.zeros_like(param.data, dtype=dtype))

    def update(self, param, grad):
        self.v[param.id].data += np.square(grad.data)
        grad.data *= (np.sqrt(self.v[param.id].data) + self.eps) ** -1
        param.data -= self.lr * grad.data


class AdaDelta(Optimizer):

    def __init__(self, rho=0.95, eps=1e-06):
        self.rho = rho
        self.eps = eps
        self.v = {}
        self.s = {}

    def init_param(self, param, dtype):
        if param.id not in self.v:
            self.v[param.id] = Parameter(
                param.id, np.zeros_like(param.data, dtype=dtype))
        if param.id not in self.s:
            self.s[param.id] = Parameter(
                param.id, np.zeros_like(param.data, dtype=dtype))

    def update(self, param, grad):
        self.v[param.id].data *= self.rho
        self.v[param.id].data += (1 - self.rho) * np.square(grad.data)
        grad.data *= np.sqrt(self.s[param.id].data + self.eps)
        grad.data *= np.sqrt(self.v[param.id].data + self.eps) ** -1
        self.s[param.id].data *= self.rho
        self.s[param.id].data += (1 - self.rho) * np.square(grad.data)
        param.data -= grad.data


class Adam(Optimizer):

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 1
        self.m = {}
        self.v = {}

    def init_param(self, param, dtype):
        if param.id not in self.m:
            self.m[param.id] = Parameter(
                param.id, np.zeros_like(param.data, dtype=dtype))
        if param.id not in self.v:
            self.v[param.id] = Parameter(
                param.id, np.zeros_like(param.data, dtype=dtype))

    def update(self, param, grad):
        self.m[param.id].data *= self.beta1
        self.m[param.id].data += (1 - self.beta1) * grad.data
        self.v[param.id].data *= self.beta2
        self.v[param.id].data += (1 - self.beta2) * np.square(grad.data)
        m_corr = self.m[param.id].data * ((1 - self.beta1 ** self.t) ** -1)
        v_corr = self.v[param.id].data * ((1 - self.beta2 ** self.t) ** -1)
        self.t += 1
        param.data -= self.alpha * m_corr * ((np.sqrt(v_corr)+self.eps) ** -1)
