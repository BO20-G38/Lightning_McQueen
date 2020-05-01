import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


from cars.load.dataset import load_x_dataset_m_1, load_y_dataset_m_1
from cars.plotting.plot_graph import plot_model
from cars.testing.test_mcqueen import test_model

from keras.applications import MobileNet
from keras.optimizers import Adam


# Loading dataset
X = load_x_dataset_m_1()
y = load_y_dataset_m_1()

NAME = '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_pase_two/McQueen'  # Model name

model = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet',
                  input_tensor=None, pooling=None, classes=1000)

model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=30, validation_split=0.3, verbose=2)

plot_model(history, 'acc', 'McQueen_Phase_Two_CNN', '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_pase_two/acc.png')
plot_model(history, 'loss', 'McQueen_Phase_Two_CNN', '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_pase_two/loss.png')

model.save(NAME + '.model')  # Saving the model
test_model(NAME+'.model')  # Testing the model with testset1
