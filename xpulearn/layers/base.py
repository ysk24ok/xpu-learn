from abc import abstractmethod


class Layer(object):

    @abstractmethod
    def forwardprop(self):
        pass

    @abstractmethod
    def backprop(self):
        pass
