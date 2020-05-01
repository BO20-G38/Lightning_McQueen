# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Sets PlaidML as backend with Keras on top.
# Proceeds then to load a given model, specify early stop
# callback for escaping overfittment. Then proceeds to train
# the given model for the specified amount of epochs, give
# a summary of the model then save the model and return the
# history object created by the fit() function call.
# ------------------------------------------------------- #
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam


# Train the given model
def train_model(X, y, epochs, batch_size, val_split, patience, model, save_location):
    # Load the model from the given location
    model = load_model(model)  # Load model
    # Early stop callback to escape overfitting
    early_stop = EarlyStopping(monitor="val_loss",
                               min_delta=0,
                               patience=patience,
                               verbose=0,
                               mode="auto",
                               baseline=None,
                               restore_best_weights=False)  # Stops the training early of the val_loss has stopped improving

    model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.0001, momentum=0.6), metrics=['accuracy'])

    # Train model with early stop callback
    history = model.fit(X, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=val_split,
                        callbacks=[early_stop])  # train the model

    # Prints model summary & save model to given location
    model.summary()
    model.save(save_location)

    # Return history object
    return history

