from . import xp


def accuracy_score(Y_true, Y_pred):
    """
    Arguments:
        y_true (numpy.ndarray or cupy.core.core.ndarray): 2D array
        y_pred (numpy.ndarray or cupy.core.core.ndarray): 2D array
    Returns:
        score (numpy.float32)
    """
    T = xp.argmax(Y_true, axis=1) == xp.argmax(Y_pred, axis=1)
    return xp.float(xp.sum(T) / Y_true.shape[0])
