import numpy
from keras.datasets import mnist
from keras.utils import np_utils

from xpulearn import optimizers, xp
from xpulearn.layers import Activation, Dense, Dropout
from xpulearn.metrics import accuracy_score
from xpulearn.model import Model

batch_size = 128
num_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

# When using CuPy
if xp != numpy:
    X_train = xp.asarray(X_train)
    Y_train = xp.asarray(Y_train)
    X_test = xp.asarray(X_test)
    Y_test = xp.asarray(Y_test)

model = Model(input_shape=(784,))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')
model.fit(X_train, Y_train, epochs=20, verbose=True)
print('Test accuracy: {:.2f} %'.format(
    accuracy_score(Y_test, model.predict(X_test)) * 100))
