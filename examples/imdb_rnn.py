import numpy
from keras.preprocessing import sequence
from keras.datasets import imdb

from xpulearn import optimizers, xp
from xpulearn.layers import Dense, Embedding, RNN
from xpulearn.metrics import accuracy_score
from xpulearn.model import Model

max_features = 20000
maxlen = 80

print('Loading data...')
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)

# When using CuPy
if xp != numpy:
    X_train = xp.asarray(X_train)
    Y_train = xp.asarray(Y_train)
    X_test = xp.asarray(X_test)
    Y_test = xp.asarray(Y_test)

print('Build model...')
model = Model(input_dim=None)
model.add(Embedding(max_features, 128))
model.add(RNN(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam())

print('Train...')
model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=True)
print('Test accuracy: {:.2f} %'.format(
    accuracy_score(Y_test, model.predict(X_test)) * 100))
