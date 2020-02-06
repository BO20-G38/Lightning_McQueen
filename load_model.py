import os
import numpy as np
import pickle

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
import matplotlib.pyplot as plt
from keras.utils import normalize

pickle_in = open("/Users/william/Documents/gitHub/B20IT38/label_dataset/X2.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("/Users/william/Documents/gitHub/B20IT38/label_dataset/y2.pickle", "rb")
y = pickle.load(pickle_in)

X = normalize(X, axis=1)
y_label = np.asarray(y)

model = load_model("/Users/william/Documents/gitHub/B20IT38/plaidml_test_model/models/FF-NN-1-test.model")

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy', # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

history = model.fit(X, y_label, epochs=50, validation_split=0.3) # train the model

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('FF-NN-1-test_Model_accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('FF-NN-1-test-1_Model_loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


model.summary()
model.save('models/retrained/FF-NN-1-test-1.model')
