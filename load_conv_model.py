import os
import numpy as np
import pickle

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
import matplotlib.pyplot as plt
from keras.utils import normalize
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Load dataset
pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_1/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_1/y.pickle", "rb")
y = pickle.load(pickle_in)

# Normalize data
X = X/255.0
y_label = np.asarray(y)


model = load_model("/Users/william/Documents/gitHub/B20IT38/plaidml_test_model/models/base_conv_model/conv2d-1-64x64-training_d_1.model")  # Load model
checkpointer = ModelCheckpoint(filepath='/Users/william/Documents/gitHub/B20IT38/plaidml_test_model/models/base_conv_model/retrain_1_d_1/weights/weight_save.hdf5', verbose=1, save_best_only=True)  # saves the models weights after each epoch if the validation loss decreased
early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=3, verbose=0, mode="auto", baseline=None, restore_best_weights=False)  # Stops the training early of the val_loss has stopped improving

history = model.fit(X, y_label, epochs=6, batch_size=200, validation_split=0.3, callbacks=[checkpointer, early_stop])  # train the model

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
model.save('/Users/william/Documents/gitHub/B20IT38/plaidml_test_model/models/base_conv_model/retrain_1_d_1/conv2d_retrain1_d1.model')
