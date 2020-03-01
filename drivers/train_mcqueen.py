# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Loading dataset, training and plotting given model
# ------------------------------------------------------- #

from cars.plotting.plot_graph import plot_model
from cars.load.dataset import load_x_dataset2, load_y_dataset2
from cars.load.model import train_model
from cars.testing.test_mcqueen import test_model


NAME = 'McQueen_d2'  # name for the new model
FOLDER = '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/models/retrain_1_d2/'  # Folder that saves are going

# Load dataset
X = load_x_dataset2()
y = load_y_dataset2()

# Load model, create early stop callback & train model
history = train_model(X=X, y=y, epochs=2, batch_size=150, val_split=0.2, patience=2,
                      model="/Users/william/Documents/gitHub/B20IT38/mcqueen_models/models/base/McQueen.model",
                      save_location=FOLDER+NAME+'.model')

# Plot training & validation acc & loss
plot_model(history=history, metric='acc', name=NAME, save_location=FOLDER+'acc.png')
plot_model(history=history, metric='loss', name=NAME, save_location=FOLDER+'loss.png')

# Test the trained model against the testing set
# Function prints how many correct predictions that was made.
test_model(FOLDER+NAME+'.model')
