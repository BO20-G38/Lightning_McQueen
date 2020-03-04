import os
from cars.load.dataset import load_y_dataset1, load_x_dataset1
from cars.plotting.plot_graph import plot_model
from cars.testing.test_mcqueen import test_model
from cars.plotting.plot_model import visualize_model

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import normalize
from keras.callbacks import ModelCheckpoint, EarlyStopping

NAME = '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_FF/base/McQueen'  # Model name

# Loading dataset
X = load_x_dataset1()
y = load_y_dataset1()

# Model definition
model = Sequential()  # a basic feed-forward model
model.add(Flatten())
model.add(Dense(512, activation='relu', input_shape=X.shape[1:]))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Stops the training early of the val_loss has stopped improving
early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=7, verbose=0,
                           mode="auto", baseline=None, restore_best_weights=False)

# arch = visualize_model(model, 'model.png')

# Trains the model and saves the history of the training
history = model.fit(X, y, batch_size=150, epochs=2, validation_split=0.2, callbacks=[early_stop])

# Plotting the acc & loss of the training and validation
plot_model(history, 'acc', 'McQueen_Fully_connected', '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_FF/base/acc.png')
plot_model(history, 'loss', 'McQueen_Fully_connected', '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_FF/base/loss.png')


model.save(NAME + '.model')  # Saving the model
test_model(NAME+'.model')  # Testing the model with testset1