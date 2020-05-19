# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 16.02.2020
# plot images from given dataset
# ------------------------------------------------------- #
import matplotlib.pyplot as plt


def show_img(index, x):
    # x = load_x_dataset1()
    plt.imshow(x[index].reshape(64, 64), cmap='gray')
    plt.show()
