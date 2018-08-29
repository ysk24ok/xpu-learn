from abc import abstractmethod


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
