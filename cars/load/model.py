import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping


def train_model(X, y, epochs, batch_size, val_split, patience, model, save_location):
    model = load_model(model)  # Load model
    early_stop = EarlyStopping(monitor="val_loss",
                               min_delta=0,
                               patience=patience,
                               verbose=0,
                               mode="auto",
                               baseline=None,
                               restore_best_weights=False)  # Stops the training early of the val_loss has stopped improving

    history = model.fit(X, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=val_split,
                        callbacks=[early_stop])  # train the model

    model.summary()
    model.save(save_location)

    return history





#checkpointer = ModelCheckpoint(filepath='/plaidml_test_model/models/McQueen/weight_save.hdf5', verbose=1, save_best_only=True)  # saves the models weights after each epoch if the validation loss decreased
