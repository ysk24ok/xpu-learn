from . import xp


def accuracy_score(Y_true, Y_pred):
    """
    Arguments:
        Y_true (2D or 1D array): True labels
        Y_pred (2D array): Predicted labels
    Returns:
        score (xp.float32)
    """
    if len(Y_true.shape) == 2:
        T = xp.argmax(Y_true, axis=1) == xp.argmax(Y_pred, axis=1)
    else:
        T = Y_true == (Y_pred.flatten() >= 0.5)
    return xp.float(xp.sum(T) / Y_true.shape[0])
