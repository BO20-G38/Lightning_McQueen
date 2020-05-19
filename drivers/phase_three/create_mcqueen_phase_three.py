import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from cars.load.dataset import load_y_2, load_x_2
from cars.plotting.plot_training import plot_model

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping

FOLDER = "E:/models4/t1/"
NAME = FOLDER + 'McQueen_p3'  # Model name

# Loading dataset
X = load_x_2()
y = load_y_2()

# Model definition
model = Sequential()
model.add(Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(40, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(700, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(350, activation='relu'))
model.add(Dense(6, activation='softmax'))

early_stop = EarlyStopping(monitor="val_loss",
                           min_delta=0,
                           patience=2,
                           verbose=0,
                           mode="auto",
                           baseline=None,
                           restore_best_weights=True)

model.compile(optimizer=SGD(lr=0.001, momentum=0.8, decay=1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trains the model and saves the history of the training
history = model.fit(X, y, batch_size=200, epochs=10, validation_split=0.3, callbacks=[early_stop])

model.summary()
# Plotting the acc & loss of the training and validation
plot_model(history, 'acc', 'McQueen_p3_t1', FOLDER+'acc.png')
plot_model(history, 'loss', 'McQueen_P3_t1', FOLDER+'loss.png')

model.save(NAME + '.model')  # Saving the model
# test_model(NAME+'.model')  # Testing the model with testset1
