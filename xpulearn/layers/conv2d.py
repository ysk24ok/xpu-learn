from .. import xp, Parameter
from .base import Layer


def im2col(img, kernel_h, kernel_w, stride=1, padding=0):
    """Converts image to column.
    For example, if img.shape is (1, 3, 4, 4), kernel_h is 3, kernel_w is 3,
    this function converts
           channel=1      channel=2      channel=3
        [[ 0  1  2  3], [16 17 18 19], [32 33 34 35]
         [ 4  5  6  7], [20 21 22 23], [36 37 38 39]
         [ 8  9 10 11], [24 25 26 27], [40 41 42 43]
         [12 13 14 15], [28 29 30 31], [44 45 46 47]]
    into
        [[ 0  1  2  4  5  6  8  9 10
          16 17 18 20 21 22 24 25 26
          32 33 34 36 37 38 40 41 42]
         [ 1  2  3  5  6  7  9 10 11
          17 18 19 21 22 23 25 26 27
          33 34 35 37 38 39 41 42 43]
         [ 4  5  6  8  9 10 12 13 14
          20 21 22 24 25 26 28 29 30
          36 37 38 40 41 42 44 45 46]
         [ 5  6  7  9 10 11 13 14 15
          21 22 23 25 26 27 29 30 31
          37 38 39 41 42 43 45 46 47]]

    Arguments:
        img (numpy.ndarray or cupy.core.core.ndarray): 4D array of shape
            [batch size, input channels, input height, input width]
        kernel_h (int): The height of a kernel
        kernel_w (int): The width of a kernel
        stride (int): stride
        padding (int): padding
    Returns:
        (numpy.ndarray or cupy.core.core.ndarray): 2D array of shape
            [batch size * output height * output width,
             input channels * kernel height * kernel width]
    """
    batch_size, input_channels, input_h, input_w = img.shape
    output_h = (input_h + 2 * padding - kernel_h) // stride + 1
    output_w = (input_w + 2 * padding - kernel_w) // stride + 1

    padded_img = xp.pad(img, [
        (0,0), (0,0), (padding, padding), (padding, padding)], 'constant')
    col = xp.zeros((batch_size, input_channels, kernel_h, kernel_w, output_h, output_w))

    for y in range(kernel_h):
        y_max = y + stride * output_h
        for x in range(kernel_w):
            x_max = x + stride * output_w
            col[:, :, y, x, :, :] = padded_img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * output_h * output_w, -1)
    return col


def col2im(col, input_shape, kernel_h, kernel_w, stride=1, padding=0):
    """Converts column to image.

    Arguments:
        col: (numpy.ndarray or cupy.core.core.ndarray): 2D array of shape
            [batch size * output height * output width,
             input channels * kernel height * kernel width]
        input_shape (tuple[int]): input shape
            [batch size, input channels, input height, input width]
        kernel_h (int): The height of a kernel
        kernel_w (int): The width of a kernel
        stride (int): stride
        padding (int): padding
    Returns:
        (numpy.ndarray or cupy.core.core.ndarray): 4D array of shape
            [batch size, input channels, input height, input width]
    """
    batch_size, input_channels, input_h, input_w = input_shape
    output_h = (input_h + 2 * padding - kernel_h) // stride + 1
    output_w = (input_w + 2 * padding - kernel_w) // stride + 1
    col = col.reshape(batch_size, output_h, output_w, input_channels, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)

    padded_input_h = input_h + 2 * padding + stride - 1
    padded_input_w = input_w + 2 * padding + stride - 1
    img = xp.zeros((batch_size, input_channels, padded_input_h, padded_input_w))
    for y in range(kernel_h):
        y_max = y + stride * output_h
        for x in range(kernel_w):
            x_max = x + stride * output_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, padding:input_h + padding, padding:input_w + padding]


class Conv2d(Layer):

    def __init__(self, in_channels, output_channels, kernel_size,
                 stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.kernel_size = kernerl_size
        self.stride = stride
        self.padding = padding
        self.params = {}
        self.grads = {}

    def init_params(self, base_layer_id, _, dtype):
        if 'W' not in self.params and 'W' not in self.grads:
            layer_id = '{}_W'.format(base_layer_id)
            shape = (self.output_channels, self.in_channels,
                     self.kernel_size[0], self.kernel_size[1])
            self.params['W'] = Parameter(
                layer_id, xp.random.randn(*shape).astype(dtype))
        if 'b' not in self.params and 'b' not in self.grads:
            layer_id = '{}_b'.format(base_layer_id)
            shape = (self.output_channels,)
            self.params['b'] = Parameter(
                layer_id, xp.random.randn(*shape).astype(dtype))

    def forwardprop(self, X_in):
        """
        Arguments:
            X_in (numpy.ndarray or cupy.core.core.ndarray): 4D array of shape
                [batch size, input channels, height, width]
        Returns:
            (numpy.ndarray or cupy.core.core.ndarray): 4D array of shape
                [batch size, output channels, height, width]
        """
        batch_size, _, input_h, input_w = X_in.shape
        output_h = int(1 + (input_h + 2 * self.padding - self.kernel_size[0]) / self.stride)
        output_w = int(1 + (input_w + 2 * self.padding - self.kernel_size[1]) / self.stride)
        col = im2col(X_in, kernel_h, kernel_w, self.stride, self.padding)
        col_W = self.params['W'].data.reshape(self.output_channels, -1).T
        out = col @ col_W + self.params['b'].data
        out = out.reshape(batch_size, output_h, output_w, -1).transpose(0, 3, 1, 2)
        return out

    def backprop(self, dout):
        pass
