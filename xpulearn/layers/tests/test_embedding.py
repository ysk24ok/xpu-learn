import numpy as np
import pytest

from xpulearn.layers import Embedding


@pytest.fixture
def fixtures():
    # 0: padded, 1: BOS, 2: OOV
    X = np.array([
        [0, 0, 0, 0, 0, 1, 2, 5],
        [0, 0, 1, 7, 4, 2, 8, 4],
        [1, 9, 2, 6, 7, 2, 3, 2],
        [0, 0, 0, 1, 2, 4, 2, 8]
    ])
    vocab_size = 10
    embedding_size = 5
    layer = Embedding(vocab_size, embedding_size)
    layer.init_params(1, None, 'float32')
    return layer, X


def test_init_params(fixtures):
    layer, _ = fixtures
    assert layer.params['W'].data.shape == (layer.vocab_size, layer.num_units)
    assert layer.grads['W'].data.shape == (layer.vocab_size, layer.num_units)


def test_forwardprop(fixtures):
    layer, X = fixtures
    batch_size = X.shape[0]
    timesteps = X.shape[1]
    got = layer.forwardprop(X)
    assert got.shape == (batch_size, timesteps, layer.num_units)


def test_backprop(fixtures):
    layer, X = fixtures
    batch_size = X.shape[0]
    timesteps = X.shape[1]
    layer.forwardprop(X)
    dX = np.random.randn(batch_size, timesteps, layer.num_units)
    layer.backprop(dX)
    np.testing.assert_array_almost_equal(
        layer.grads['W'].data[4], dX[1][4] + dX[1][7] + dX[3][5])
