from . import xp


def clip_before_exp(X, dtype):
    if dtype == 'float64':
        max_ = 700
    elif dtype == 'float32':
        max_ = 85
    else:
        raise ValueError('Invalid value for dtype: {}'.format(dtype))
    return xp.clip(X, None, max_)


def clip_before_log(X, dtype):
    if dtype == 'float64':
        eps_ = xp.finfo(xp.float64).eps
    elif dtype == 'float32':
        eps_ = xp.finfo(xp.float32).eps
    else:
        raise ValueError('Invalid value for dtype: {}'.format(dtype))
    return xp.clip(X, eps_, 1-eps_)
