# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Loading dataset, training and plotting given model
# ------------------------------------------------------- #

from cars.plotting.plot_graph import plot_model
from cars.load.dataset import load_x_dataset1, load_y_dataset1
from cars.load.model import train_model

# Model name
NAME = 'McQueen_test'

# Load dataset
X = load_x_dataset1()
y = load_y_dataset1()

# Load model, create early stop callback & train model
history = train_model(X=X, y=y, epochs=1, batch_size=150, val_split=0.3, patience=3,
                        model="models/mcqueen_base/mcqueen_retrained_2_d2/McQueen_d2.model",
                        save_location='models/McQueen/' + NAME + '.model')

# Plot training & validation acc & loss
plot_model(history=history, metric='acc', name=NAME, save_location='models/McQueen/acc.png')
plot_model(history=history, metric='loss', name=NAME, save_location='models/McQueen/loss.png')