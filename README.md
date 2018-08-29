# xpu-learn

This repository is my personal project that implements deep learning framework from scratch using [NumPy](http://www.numpy.org/) and [CuPy](https://cupy.chainer.org/).

## Running on CPU

## Running on GPU

### Google Colaboratory

First, enables GPU on the notebook by changing `Runtime` -> `Change runtime type` -> `Hardware accelerator` from `None` to `GPU`.

Next, install CuPy.

```
!xpu-learn/setup_cupy_on_google_colab.sh
```

Clone this repository.

```
!git clone -b use_cupy https://github.com/ysk24ok/xpu-learn.git
```

Run example code.

```
!PYTHONPATH=./xpu-learn python3 xpu-learn/examples/mnist_mlp.py
```
