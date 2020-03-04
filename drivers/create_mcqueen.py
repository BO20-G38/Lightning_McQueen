# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Initial creation of the McQueen model for recognising
# arm gestures.
# ------------------------------------------------------- #
import os
from cars.plotting.plot_graph import plot_model
from cars.load.dataset import load_x_dataset1, load_y_dataset1

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD


# TODO: change file paths!!!!
NAME = "models/mcqueen_phase_two/base/mcqueen"  # Model name

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

model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
history = model.fit(X, y, batch_size=150, epochs=6, verbose=1, validation_split=0.3, callbacks=[early_stop])

# Plot training & validation accuracy & loss values
plot_model(history, 'acc', 'McQueen', 'models/mcqueen_phase_two/base/acc.png')
plot_model(history, 'loss', 'McQueen', 'models/mcqueen_phase_two/base/loss.png')

model.summary()
model.save(NAME + '.model')

