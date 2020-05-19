import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
import cv2
import numpy as np


CATEGORIES = ["forward", "right", "left", "backward", "stop", "still"]


def prepare(filepath):
    IMG_SIZE = 100  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    print(new_array)
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255  # return the image with shaping that the model wants.


def pred_img(img_path):
    model = load_model('E:/models3/t12/McQueen_p3.model')
    pred = model.predict_classes(prepare(img_path))
    classes = np.argmax(pred[0], axis=0)
    print(CATEGORIES[classes])


pred_img('E:/models3/imgs/stop.jpg')
