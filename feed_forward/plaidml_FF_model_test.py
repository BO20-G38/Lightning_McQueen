import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import normalize
from keras.callbacks import ModelCheckpoint, EarlyStopping

NAME = "models/base_FF_model/FF-NN-1-64x64-training_d_1"  # Model name

# Loading dataset
pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_1/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_1/y.pickle", "rb")
y = pickle.load(pickle_in)

# X = X/255.0
# Normalizing data
X = normalize(X, axis=1)
y_label = np.asarray(y)

# Model definition
model = Sequential()  # a basic feed-forward model
model.add(Flatten())  # takes our 100x100 and makes it 1x10000
model.add(Dense(512, activation='relu', input_shape=X.shape[1:], name="dense_1"))
model.add(Dense(512, activation='relu', name="dense_2"))  # a simple fully-connected layer, 128 units, relu activation
model.add(Dense(512, activation='relu', name="dense_3"))  # a simple fully-connected layer, 128 units, relu activation
model.add(Dense(5, activation='softmax', name="dense_output"))  # our output layer. 5 units for 5 classes. Softmax for probability distribution

# model.load_weights("/Users/william/Documents/gitHub/B20IT38/plaidml_test_model/models/base_FF_model/weight_saves/weight_save.hdf5", by_name=True)

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

checkpointer = ModelCheckpoint(filepath='models/base_FF_model/weight_saves/weight_save2.hdf5', verbose=1, save_best_only=True)  # saves the models weights after each epoch if the validation loss decreased
early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=15, verbose=0, mode="auto", baseline=None, restore_best_weights=False)  # Stops the training early of the val_loss has stopped improving
history = model.fit(X, y_label, batch_size=32, epochs=100, validation_split=0.3, callbacks=[checkpointer, early_stop])  # Trains the model and saves the history of the training

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
