# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# plot images from given dataset
# ------------------------------------------------------- #
import matplotlib.pyplot as plt
from cars.load.dataset import load_x_dataset1


# Show an image from your dataset if the image 1x....
def show_linear_img(index):
    # Loading dataset
    x = load_x_dataset1()

    # Show image
    plt.imshow(x[index].reshape(64, 64), cmap='gray')
    plt.show()


def show_img(index, x):
    # x = load_x_dataset1()
    plt.imshow(x[index].reshape(64, 64), cmap='gray')
    plt.show()
