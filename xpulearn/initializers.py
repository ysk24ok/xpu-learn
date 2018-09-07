from abc import abstractmethod

from . import xp


class Initializer(object):

    def __init__(self, initializer, dtype):
        if initializer == 'zeros':
            self.initializer = Zeros(dtype)
        elif initializer == 'randn':
            self.initializer = Randn(dtype)
        elif initializer == 'he':
            self.initializer = He(dtype)
        else:
            msg = 'Invalid value for initializer: {}'.format(initializer)
            raise ValueError(msg)

    def __call__(self, shape):
        return self.initializer(shape)


class BaseInitializer(object):

    def __init__(self, dtype):
        self.dtype = dtype

    @abstractmethod
    def __call__(self, shape):
        pass


class Zeros(BaseInitializer):

    def __call__(self, shape):
        return xp.zeros(shape, dtype=self.dtype)


class Randn(BaseInitializer):

    def __call__(self, shape):
        return xp.random.randn(*shape).astype(self.dtype) / 10


class He(BaseInitializer):

    def __call__(self, shape):
        factor = xp.sqrt(2 / shape[-1])
        return xp.random.randn(*shape).astype(self.dtype) * factor
