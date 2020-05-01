# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 01.03.2020
# test with testset and write to log
# ------------------------------------------------------- #
from keras.models import load_model
from cars.load.dataset import load_x_testset_m_2, load_y_testset_m_2
from cars.matrix.ConfusionMatrix import ConfusionMatrix


def test_model(model_name):

    #        forward    right    left    backward    stop    still
    categories = ['forward', 'right', 'left', 'backward', 'stop', 'still']  # categories

    x_test = load_x_testset_m_2()  # load dataset
    y_test = load_y_testset_m_2()  # load dataset

    print('Loading model: ' + model_name)
    model = load_model(model_name)  # load the model
    print('Testing model!')
    y_new = model.predict_classes(x_test)  # Prediction

    matrix = ConfusionMatrix(categories)

    for i in range(len(x_test)):  # check the predictions
        matrix.add_pair(predicted_category=categories[y_new[i]], actual_category=categories[y_test[i]])

    print(matrix)

    # Prints loss function first and then acc on the testset
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)


# Writes result of testing to log file
def open_text_file(log):
    file = open('/Users/william/Documents/gitHub/B20IT38/mcqueen_models/test_log.txt', 'a')
    file.write(log)
    file.close()




