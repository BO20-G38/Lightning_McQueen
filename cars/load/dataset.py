# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Load datasets into project
# ------------------------------------------------------- #
import pickle
import numpy as np


# Load dataset 1
def load_x_dataset1():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_1/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/255.0
    return x


def load_y_dataset1():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_1/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


# Load dataset 2
def load_x_dataset2():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_2/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x / 255.0
    return x


def load_y_dataset2():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_2/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_dataset3():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_3/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/255.0
    return x


def load_y_dataset3():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_3/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


# Load testset 1
def load_x_testset1():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/t_1/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x / 255.0
    return x


def load_y_testset1():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/t_1/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y
