import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Arguments:
        y_true (numpy.ndarray): 1D array
        y_pred (numpy.ndarray): 1D array
    Returns:
        score (float)
    """
    return np.sum(y_true == y_pred) / y_true.shape[0]
