from unittest import TestCase

from xpulearn import xp
from xpulearn.layers import Conv2d
from xpulearn.layers.conv2d import im2col, col2im


class TestConvUtils(TestCase):

    def _test_im2col(self, batch_size, input_channels, input_h, input_w,
                     kernel_h, kernel_w, stride, padding):
        im = xp.arange(batch_size * input_channels * input_h * input_w).reshape(
            (batch_size, input_channels, input_h, input_w))
        col = im2col(im, kernel_h, kernel_w, stride=stride, padding=padding)
        output_h = (input_h + 2 * padding - kernel_h) // stride + 1
        output_w = (input_w + 2 * padding - kernel_w) // stride + 1
        assert col.shape[0] == batch_size * output_h * output_w
        assert col.shape[1] == input_channels * kernel_h * kernel_w
        row_idx = 0
        for batch_idx in range(batch_size):
            for initial_h_idx in range(-padding, input_h - kernel_h + padding + 1, stride):
                for initial_w_idx in range(-padding, input_w - kernel_w + padding + 1, stride):
                    col_idx = 0
                    for input_channel_idx in range(input_channels):
                        for kernel_h_idx in range(kernel_h):
                            input_h_idx = initial_h_idx + kernel_h_idx
                            for kernel_w_idx in range(kernel_w):
                                input_w_idx = initial_w_idx + kernel_w_idx
                                #print(row_idx, col_idx, batch_idx, input_channel_idx, input_h_idx, input_w_idx)
                                if 0 <= input_h_idx and input_h_idx < input_h and 0 <= input_w_idx and input_w_idx < input_w:
                                    xp.testing.assert_almost_equal(
                                        col[row_idx][col_idx],
                                        im[batch_idx][input_channel_idx][input_h_idx][input_w_idx])
                                else:
                                    xp.testing.assert_almost_equal(col[row_idx][col_idx], 0.0)
                                col_idx += 1
                    row_idx += 1

    def _test_col2im(self, batch_size, input_channels, input_h, input_w,
                     kernel_h, kernel_w, stride, padding):
        input_shape = (batch_size, input_channels, input_h, input_w)
        im = xp.arange(batch_size * input_channels * input_h * input_w).reshape(input_shape)
        output_h = (input_h + 2 * padding - kernel_h) // stride + 1
        output_w = (input_w + 2 * padding - kernel_w) // stride + 1
        col = im2col(im, kernel_h, kernel_w, stride=stride, padding=padding)
        print(col)
        got = col2im(col, input_shape, kernel_h, kernel_w,
                     stride=stride, padding=padding)
        print(got)
        assert got.shape[0] == input_shape[0]
        assert got.shape[1] == input_shape[1]
        assert got.shape[2] == input_shape[2]
        assert got.shape[3] == input_shape[3]
        #xp.testing.assert_array_almost_equal(got, im)

    def test_im2col_when_stride_is_1_and_padding_is_0(self):
        batch_size, input_channels, input_h, input_w = 4, 3, 5, 5
        kernel_h, kernel_w, stride, padding = 3, 3, 1, 0
        self._test_im2col(batch_size, input_channels, input_h, input_w,
                          kernel_h, kernel_w, stride, padding)

    def test_im2col_when_stride_is_2_and_padding_is_0(self):
        batch_size, input_channels, input_h, input_w = 4, 3, 5, 5
        kernel_h, kernel_w, stride, padding = 3, 3, 2, 0
        self._test_im2col(batch_size, input_channels, input_h, input_w,
                          kernel_h, kernel_w, stride, padding)

    def test_im2col_when_stride_is_1_and_padding_is_1(self):
        batch_size, input_channels, input_h, input_w = 4, 3, 5, 5
        kernel_h, kernel_w, stride, padding = 3, 3, 1, 1
        self._test_im2col(batch_size, input_channels, input_h, input_w,
                          kernel_h, kernel_w, stride, padding)

    def test_im2col_when_stride_is_1_and_padding_is_2(self):
        batch_size, input_channels, input_h, input_w = 4, 3, 5, 5
        kernel_h, kernel_w, stride, padding = 3, 3, 1, 2
        self._test_im2col(batch_size, input_channels, input_h, input_w,
                          kernel_h, kernel_w, stride, padding)

    def test_im2col_when_stride_is_2_and_padding_is_2(self):
        batch_size, input_channels, input_h, input_w = 4, 3, 5, 5
        kernel_h, kernel_w, stride, padding = 3, 3, 2, 2
        self._test_im2col(batch_size, input_channels, input_h, input_w,
                          kernel_h, kernel_w, stride, padding)

    def test_col2im_when_stride_is_1_and_padding_is_0(self):
        batch_size, input_channels, input_h, input_w = 4, 3, 5, 5
        kernel_h, kernel_w, stride, padding = 3, 3, 1, 0
        self._test_col2im(batch_size, input_channels, input_h, input_w,
                          kernel_h, kernel_w, stride, padding)


class TestConv2d(object):
    pass
