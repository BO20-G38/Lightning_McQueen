# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Load datasets into project
# ------------------------------------------------------- #
import pickle
import numpy as np


# ------- MobileNet Dataset ----------
def load_x_1():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/Dataset_1_room/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/225
    return x


def load_y_1():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/Dataset_1_room/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_2():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/Dataset_2_livingroom/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/225
    return x


def load_y_2():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/Dataset_2_livingroom/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_3():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/dataset_4_auditoriet/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/225
    return x


def load_y_3():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/dataset_4_auditoriet/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_4():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/dataset_5_forest/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/225
    return x


def load_y_4():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/dataset_5_forest/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_5():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/dataset_7_street/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/225
    return x


def load_y_5():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/dataset_7_street/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y
# ------------------------------------


# For mobileNet model -------
def load_x_testset_m_1():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/Dataset_3_white/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/225
    return x


def load_y_testset_m_1():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/Dataset_3_white/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_testset_m_2():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/Dataset_6_complex/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/225
    return x


def load_y_testset_m_2():
    pickle_in = open("/Users/william/Documents/gitHub/B20IT38/greenscreen_data/Dataset_6_complex/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y

