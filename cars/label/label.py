import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

CATEGORIES = ["forward", "right", "left", "backward", "stop", "still"]
training_data = []
IMG_SIZE = 224

def create_training_data(DATADIR):
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass

    random.shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    pickle_out = open(DATADIR + "/X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open(DATADIR + "/y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


create_training_data('/Users/william/Documents/gitHub/B20IT38/greenscreen_data/dataset_1_room')


