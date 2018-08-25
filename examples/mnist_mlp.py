import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from xpulearn import optimizers
from xpulearn.layers import Activation, Dense
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

print('X_train shape: {}'.format(X_train.shape))
print('Y_train shape: {}'.format(Y_train.shape))

model = Model(
    loss='categorical_crossentropy', input_shape=(784,),
    optimizer=optimizers.AdaDelta())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.fit(X_train, Y_train, epochs=20, verbose=True)
y_test_pred = np.argmax(model.predict(X_test), axis=1)
print('Test accuracy: {:.2f} %'.format(
    accuracy_score(y_test, y_test_pred) * 100))
