# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 01.03.2020
# test with testset and write to log
# ------------------------------------------------------- #
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
from cars.load.dataset import load_x_testset1, load_y_testset1


def test_model(model_name):

    # CATEGORIES = ["forward", "right", "left", "backward", "stop"]  # categories
    n = 0  # counter

    x_test = load_x_testset1()  # load dataset
    y_test = load_y_testset1()  # load dataset

    model = load_model(model_name)  # load the model
    y_new = model.predict_classes(x_test)  # Prediction

    for i in range(len(x_test)):  # check the predictions
        if int(y_test[i]) == int(y_new[i]):  # If the prediction is correct
            n = n+1  # increment correct counter

    log = '-------\nModel location: ' + model_name + '\nTotal correct predictions: ' + str(n) + '/' + str(len(x_test)) \
          + "\n-------\n\n"
    print(log)  # print predictions
    open_text_file(log)


# Writes result of testing to log file
def open_text_file(log):
    file = open('/Users/william/Documents/gitHub/B20IT38/mcqueen_models/test_log.txt', 'a')
    file.write(log)
    file.close()




