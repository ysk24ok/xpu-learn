from abc import abstractmethod

from . import xp, Parameter


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
    def update_param(self, param, grad):
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
            self.update_param(param, grad)


class SGD(Optimizer):

    def __init__(self, lr=0.01):
        self.lr = lr

    def init_param(self, param, dtype):
        pass

    def update_param(self, param, grad):
        param.data -= self.lr * grad.data


class MomentumSGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def init_param(self, param, dtype):
        if param.id not in self.v:
            self.v[param.id] = Parameter(
                param.id, xp.zeros_like(param.data, dtype=dtype))

    def update_param(self, param, grad):
        v = self.v[param.id]
        v.data += (1 - self.momentum) * (grad.data - v.data)
        param.data -= self.lr * v.data


class RMSprop(Optimizer):

    def __init__(self, lr=0.01, alpha=0.99, eps=1e-08):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.s = {}

    def init_param(self, param, dtype):
        if param.id not in self.s:
            self.s[param.id] = Parameter(
                param.id, xp.zeros_like(param.data, dtype=dtype))

    def update_param(self, param, grad):
        s = self.s[param.id]
        s.data += (1 - self.alpha) * (xp.square(grad.data) - s.data)
        grad.data *= (xp.sqrt(s.data) + self.eps) ** -1
        param.data -= self.lr * grad.data


class AdaGrad(Optimizer):

    def __init__(self, lr=0.001, eps=1e-08):
        self.lr = lr
        self.eps = eps
        self.v = {}

    def init_param(self, param, dtype):
        if param.id not in self.v:
            self.v[param.id] = Parameter(
                param.id, xp.zeros_like(param.data, dtype=dtype))

    def update_param(self, param, grad):
        v = self.v[param.id]
        v.data += xp.square(grad.data)
        grad.data *= (xp.sqrt(v.data) + self.eps) ** -1
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
                param.id, xp.zeros_like(param.data, dtype=dtype))
        if param.id not in self.s:
            self.s[param.id] = Parameter(
                param.id, xp.zeros_like(param.data, dtype=dtype))

    def update_param(self, param, grad):
        v = self.v[param.id]
        s = self.s[param.id]
        v.data += (1 - self.rho) * (xp.square(grad.data) - v.data)
        grad.data *= xp.sqrt(s.data + self.eps)
        grad.data *= xp.sqrt(v.data + self.eps) ** -1
        s.data += (1 - self.rho) * (xp.square(grad.data) - s.data)
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
                param.id, xp.zeros_like(param.data, dtype=dtype))
        if param.id not in self.v:
            self.v[param.id] = Parameter(
                param.id, xp.zeros_like(param.data, dtype=dtype))

    def update_param(self, param, grad):
        m = self.m[param.id]
        v = self.v[param.id]
        m.data += (1 - self.beta1) * (grad.data - m.data)
        v.data += (1 - self.beta2) * (xp.square(grad.data) - v.data)
        m_hat = m.data / (1 - self.beta1 ** self.t)
        v_hat = v.data / (1 - self.beta2 ** self.t)
        self.t += 1
        param.data -= self.alpha * m_hat * ((xp.sqrt(v_hat) + self.eps) ** -1)
