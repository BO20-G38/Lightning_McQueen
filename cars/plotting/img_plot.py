import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

NAME = "models/base_model/FF-NN-1-64x64-training_d_1"  # Model name

# Loading dataset
pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_1/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("/Users/william/Documents/gitHub/B20IT38/data_and_test_sets/processed/d_1/y.pickle", "rb")
y = pickle.load(pickle_in)

plt.imshow(X[44].reshape(64, 64), cmap='gray')
plt.show()
