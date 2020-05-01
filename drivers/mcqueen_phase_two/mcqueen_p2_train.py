# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Loading dataset, training and plotting given model
# ------------------------------------------------------- #

from cars.plotting.plot_graph import plot_model
from cars.load.dataset import load_x_1, load_y_1
from cars.load.model import train_model
# from cars.testing.test_mcqueen import test_model


NAME = 'McQueen_d2'  # name for the new model
FOLDER = '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_pase_two/'  # Folder that saves are going

# Load dataset
X = load_x_1()
y = load_y_1()

# Load model, create early stop callback & train model
history = train_model(X=X, y=y, epochs=30, batch_size=150, val_split=0.3, patience=1,
                      model="/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_pase_two/McQueen.model",
                      save_location=FOLDER+NAME+'.model')

# Plot training & validation acc & loss
plot_model(history=history, metric='acc', name=NAME, save_location=FOLDER+'acc.png')
plot_model(history=history, metric='loss', name=NAME, save_location=FOLDER+'loss.png')

# Test the trained model against the testing set
# Function prints how many correct predictions that was made.
# test_model(FOLDER+NAME+'.model')
