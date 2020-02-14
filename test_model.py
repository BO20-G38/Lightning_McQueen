import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import cv2

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
from keras.utils import normalize

CATEGORIES = ["forward", "right", "left", "backward", "stop"]

forward = "/Users/william/Documents/gitHub/B20IT38/label_dataset/t1/forward/simon-forward-back-vid3-1000x1000-15fps-frame-0-bW.jpg"
backward = "/Users/william/Documents/gitHub/B20IT38/label_dataset/video/backward/ralf-backward-back-vid2-2min-1000x1000-15fps-frame-1-bW.jpg"
left = "/Users/william/Documents/gitHub/B20IT38/label_dataset/video/left/ralf-left-back-vid2-2min-1000x1000-15fps-frame-0-bW.jpg"
right = "/Users/william/Documents/gitHub/B20IT38/label_dataset/t1/right/simon-right-back-vid3-2min-1000x1000-15fps-frame-0-bW.jpg"
stop = "/Users/william/Documents/gitHub/B20IT38/label_dataset/t1/stop/simon-stop-back-vid2-2min-1000x1000-15fps-frame-0-bW.jpg"


def prepare(filepath):
    IMG_SIZE = 100  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = load_model("/Users/william/Documents/gitHub/B20IT38/plaidml_test_model/models/retrained/retrain_1/FF-NN-1-test-1.model")

prediction = model.predict([prepare(left)])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])



