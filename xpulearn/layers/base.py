from abc import abstractmethod

from .. import xp


class Layer(object):

    def __init__(self):
        self.dtype = 'float32'
        self.skip = False

    @abstractmethod
    def forwardprop(self):
        pass

    @abstractmethod
    def backprop(self):
        pass


class Dropout(Layer):

    """Dropout layer

    Parameters:
        ratio (float): ratio of dropping units in training time
        mask (numpy.ndarray or cupy.core.core.ndarray): mask array
    """

    def __init__(self, ratio):
        super(Dropout, self).__init__()
        self.ratio = ratio
        self.mask = None

    def forwardprop(self, X_in, training=False, init_mask=True):
        """
        Arguments:
            X_in (numpy.ndarray or cupy.core.core.ndarray):
                2D array of shape [batch size, #units of input-side layer]
            training (bool): units are dropped when True, default False
            init_mask (bool): initialize mask array, default True
        Returns:
            (numpy.ndarray or cupy.core.core.ndarray):
                2D array of the same shape as `X_in`
        """
        if training is False:
            return X_in
        if init_mask is True:
            self.mask = xp.random.rand(*X_in.shape) > self.ratio
        X_out = X_in * self.mask
        X_out /= (1 - self.ratio)
        return X_out

    def backprop(self, dout):
        """
        Arguments:
            dout (numpy.ndarray or cupy.core.core.ndarray):
                2D array of shape [batch size, #units of output-side layer]
        Returns:
            (numpy.ndarray or cupy.core.core.ndarray):
                2D array of the same shape as `dout`
        """
        return dout * self.mask / (1 - self.ratio)
