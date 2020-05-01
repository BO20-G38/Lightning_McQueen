# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Loading dataset, training and plotting given model
# ------------------------------------------------------- #

from cars.plotting.plot_graph import plot_model
from cars.load.dataset import load_x_3, load_y_3
from cars.load.model import train_model
from cars.testing.test_mcqueen import test_model


NAME = 'McQueen_d14'  # name for the new model
FOLDER = '/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_phase_two/training_14/'  # Folder that saves are going
OLD_MODEL = "/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_phase_two/training_13/McQueen_d13.model"  # model to train
TRAIN = True
TEST = False

if TRAIN:
    # Load dataset
    X = load_x_3()
    y = load_y_3()

    # Load model, create early stop callback & train model
    history = train_model(X=X, y=y, epochs=10, batch_size=150, val_split=0.2, patience=3, model=OLD_MODEL,
                          save_location=FOLDER+NAME+'.model')

    # Plot training & validation acc & loss
    plot_model(history=history, metric='acc', name=NAME, save_location=FOLDER+'acc_training_14.png')
    plot_model(history=history, metric='loss', name=NAME, save_location=FOLDER+'loss_training_14.png')

if TEST:
    # Test model, if test mode is true.
    test_model('/Users/william/Documents/gitHub/B20IT38/mcqueen_models/mcqueen_phase_two/training_13/McQueen_d13.model')
