import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import normalize

NAME = "models/FF-NN-1-test"

pickle_in = open("/Users/william/Documents/gitHub/B20IT38/label_dataset/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("/Users/william/Documents/gitHub/B20IT38/label_dataset/y.pickle", "rb")
y = pickle.load(pickle_in)

# X = X/255.0
X = normalize(X, axis=1)
y_label = np.asarray(y)


model = Sequential()  # a basic feed-forward model
model.add(Flatten())  # takes our 100x100 and makes it 1x10000
model.add(Dense(128, activation='relu', input_shape= X.shape[1:]))
model.add(Dense(128, activation='relu'))  # a simple fully-connected layer, 128 units, relu activation
model.add(Dense(128, activation='relu'))  # a simple fully-connected layer, 128 units, relu activation
model.add(Dense(5, activation='softmax'))  # our output layer. 5 units for 5 classes. Softmax for probability distribution

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

history = model.fit(X, y_label, batch_size=32, epochs=50, validation_split=0.3)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.summary()
model.save(NAME + '.model')
