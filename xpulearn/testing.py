import numpy as np


_epsilon = np.sqrt(np.finfo(float).eps)


def approx_fprime(xk, f, *args, epsilon=_epsilon):
    """Finite-difference approximation of the gradient of the function.
    The implementation follows that of `scipy.optimize.approx_fprime`.

    Arguments:
        xk (numpy.ndarray): 2D array at which to determine the gradient of `f`
        f (func):
            The function of which to determine the gradient.
            The return value of `f` must be the same shape as `xk` or scalar.
        args (tuple): arguments that are to be passed to `f`
        epsilon (numpy.ndarray or float): Increment to `xk`
    Returns:
        (numpy.ndarray) 2D
    """
    f0 = f(*((xk,) + args))
    grad = np.zeros(xk.shape)
    ei = np.zeros(xk.shape)
    for k1 in range(xk.shape[0]):
        for k2 in range(xk.shape[1]):
            ei[k1][k2] = 1.0
            d = epsilon * ei
            do = (f(*((xk + d,) + args)) - f0) / d[k1][k2]
            # when the output of `func` is 2D array
            if isinstance(do, np.ndarray):
                grad[k1][k2] = do[k1][k2]
            # when the output of `func` is scalar
            else:
                grad[k1][k2] = do
            ei[k1][k2] = 0.0
    return grad
