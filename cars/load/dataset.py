# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# Load datasets into project
# ------------------------------------------------------- #
import pickle
import numpy as np


def load_x_1():
    pickle_in = open("E:/pro_datasets/dataset_1_room/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/255
    return x


def load_y_1():
    pickle_in = open("E:/pro_datasets/dataset_1_room/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_2():
    pickle_in = open("E:/pro_datasets/dataset_2_livingroom/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/255
    return x


def load_y_2():
    pickle_in = open("E:/pro_datasets/dataset_2_livingroom/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_3():
    pickle_in = open("E:/pro_datasets/dataset_4_aud/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/255
    return x


def load_y_3():
    pickle_in = open("E:/pro_datasets/dataset_4_aud/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_4():
    pickle_in = open("E:/pro_datasets/dataset_5_forest/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/255
    return x


def load_y_4():
    pickle_in = open("E:/pro_datasets/dataset_5_forest/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_5():
    pickle_in = open("E:/pro_datasets/dataset_7_street/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/255
    return x


def load_y_5():
    pickle_in = open("E:/pro_datasets/dataset_7_street/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_6():
    pickle_in = open("E:/pro_datasets/dataset_8_remmen/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/255
    return x


def load_y_6():
    pickle_in = open("E:/pro_datasets/dataset_8_remmen/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_7():
    pickle_in = open("E:/pro_datasets/dataset_9_brick/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/255
    return x


def load_y_7():
    pickle_in = open("E:/pro_datasets/dataset_9_brick/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_8():
    pickle_in = open("E:/pro_datasets/dataset_10_brightness/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/255
    return x


def load_y_8():
    pickle_in = open("E:/pro_datasets/dataset_10_brightness/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y
# ------------------------------------


def load_x_testset_m_1():
    pickle_in = open("E:/pro_datasets/dataset_3_white/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/255
    return x


def load_y_testset_m_1():
    pickle_in = open("E:/pro_datasets/dataset_3_white/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y


def load_x_testset_m_2():
    pickle_in = open("E:/pro_datasets/dataset_6_complex/X.pickle", "rb")
    x = pickle.load(pickle_in)
    x = x/255
    return x


def load_y_testset_m_2():
    pickle_in = open("E:/pro_datasets/dataset_6_complex/y.pickle", "rb")
    y = pickle.load(pickle_in)
    y = np.asarray(y)
    return y

