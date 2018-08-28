from abc import abstractmethod


class Layer(object):

    def __init__(self):
        self.skip = False

    @abstractmethod
    def forwardprop(self):
        pass

    @abstractmethod
    def backprop(self):
        pass
