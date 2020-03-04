import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import pydot as pyd
from IPython.display import SVG
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt

keras.utils.vis_utils.pydot = pyd


def visualize_model(model, output):
    plot_model(model=model, to_file=output)


visualize_model()

