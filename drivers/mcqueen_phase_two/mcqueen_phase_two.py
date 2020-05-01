import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from cars.load.dataset import load_y_1, load_x_1
from cars.plotting.plot_graph import plot_model
# from cars.testing.test_mcqueen import test_model

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD


NAME = '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_pase_two/McQueen'  # Model name

# Loading dataset
X = load_x_1()
y = load_y_1()

# Model definition
model = Sequential()
model.add(Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.70))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Conv2D(40, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(900, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(350, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer=SGD(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Stops the training early of the val_loss has stopped improving
early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=3, verbose=0,
                           mode="auto", baseline=None, restore_best_weights=False)

# Trains the model and saves the history of the training
history = model.fit(X, y, batch_size=320, epochs=15, validation_split=0.2, callbacks=[early_stop])

model.summary()
# Plotting the acc & loss of the training and validation
plot_model(history, 'acc', 'McQueen_Phase_Two_CNN', '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_pase_two/acc.png')
plot_model(history, 'loss', 'McQueen_Phase_Two_CNN', '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_pase_two/loss.png')

model.save(NAME + '.model')  # Saving the model
# test_model(NAME+'.model')  # Testing the model with testset1
