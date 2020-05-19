# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Loading dataset, training and plotting given model
# ------------------------------------------------------- #

from cars.plotting.plot_training import plot_model
from cars.load.dataset import load_x_4, load_y_4
from cars.load.model import train_model
from cars.testing.test_mcqueen import test_model


NAME = 'McQueen_p3'  # name for the new model
FOLDER = 'E:/models4/t2/'  # Folder that saves are going
OLD_MODEL = "E:/models4/t1/McQueen_p3.model"  # model to train
TRAIN = True
TEST = False

if TRAIN:
    # Load dataset
    X = load_x_4()
    y = load_y_4()

    # Load model, create early stop callback & train model
    history = train_model(X=X, y=y, epochs=10, batch_size=200, val_split=0.3, patience=1, model=OLD_MODEL,
                          save_location=FOLDER+NAME+'.model')

    # Plot training & validation acc & loss
    plot_model(history=history, metric='acc', name=NAME, save_location=FOLDER+'acc.png')
    plot_model(history=history, metric='loss', name=NAME, save_location=FOLDER+'loss.png')

if TEST:
    # Test model, if test mode is true.
    test_model('E:/models3/t6/McQueen_p3.model')
