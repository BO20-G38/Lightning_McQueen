# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Initial creation of the McQueen model for recognising
# arm gestures.
# This model uses adam as optimizer
# ------------------------------------------------------- #
import os
from cars.plotting.plot_training import plot_model
from cars.load.dataset import load_x_dataset1, load_y_dataset1
from cars.testing.test_mcqueen import test_model

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping

NAME = "/Users/william/Documents/gitHub/B20IT38/mcqueen_models/models/base/mcqueen"  # Model name

# Loading dataset
X = load_x_dataset1()
y = load_y_dataset1()

# building the mode
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Stops the training early of the val_loss has stopped improving
early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=2, verbose=0, mode="auto",
                           baseline=None, restore_best_weights=False)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X, y, batch_size=150, epochs=6, verbose=1, validation_split=0.2, callbacks=[early_stop])

# Plot training & validation accuracy & loss values
plot_model(history, 'acc', 'McQueen', '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/models/mcqueen_phase_three/base/acc.png')
plot_model(history, 'loss', 'McQueen', '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/models/base/loss.png')

# Saving the model
model.save(NAME + '.model')
test_model(NAME + '.model')
