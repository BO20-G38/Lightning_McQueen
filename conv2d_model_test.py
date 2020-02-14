import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import normalize
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD

NAME = "/Users/william/Documents/gitHub/B20IT38/plaidml_test_model/models/base_conv_model/conv2d-1-64x64-training_d_1"  # Model name

# Loading dataset
pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_1/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_1/y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0
y_label = np.asarray(y)
print(X.shape)

# building the mode
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(5, activation='softmax'))

checkpointer = ModelCheckpoint(filepath='/Users/william/Documents/gitHub/B20IT38/plaidml_test_model/models/base_conv_model/weight_saves/weight_save_1.hdf5', verbose=1, save_best_only=True)  # saves the models weights after each epoch if the validation loss decreased
early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=4, verbose=0, mode="auto", baseline=None, restore_best_weights=False)  # Stops the training early of the val_loss has stopped improving

model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
history = model.fit(X, y, batch_size=200, epochs=7, verbose=1, validation_split=0.3, callbacks=[checkpointer, early_stop])

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

