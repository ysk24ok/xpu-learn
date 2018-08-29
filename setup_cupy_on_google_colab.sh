# https://github.com/kmaehashi/chainer-colab
apt -y install libcusparse8.0 libnvrtc8.0 libnvtoolsext1
ln -snf /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so.8.0 /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so
pip install cupy-cuda80
