# Made by William Svea-Lochert
# Date: 06.02.2020
import cv2
import numpy as np
import random
import os
from tqdm import tqdm


CLASS_CATEGORIES = ["forward", "right", "left", "backward", "stop", "still"]
training_data = []
dir_data = 'E:/dataset/dataset_9_brick'


def show_image(image_file):
    cv2.imshow('asd', image_file)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clean_filename(filename):
    strList = filename.split('\'')
    strList.pop()
    strList.pop(0)
    return "".join(strList)


def create_path(root, path):
    return root + '/' + path


def read_dir(path, img_count, output):
    directory = os.fsencode(path)
    convert = True

    for file in os.listdir(directory):
        filename = os.fsencode(file)
        full_file_name = create_path(path, clean_filename(str(filename)))
        if convert:
            change_bg(full_file_name, dir_data+'/' + output + '/', clean_filename(str(filename)) + '-', img_count)


def change_bg(input_path, output_path, name, image_amount):
    # Original Image
    image = cv2.imread(input_path)
    positive_roll = 220
    negative_roll = -220

    # Roll the image between -251 and 251
    if output_path == 'E:/dataset/dataset_9_brick/still' or output_path == 'E:/dataset/dataset_9_brick/stop':
        positive_roll = 200
        negative_roll = -200

    for i in tqdm(range(image_amount)):
        image_copy = np.copy(image)
        image_copy = cv2.resize(image_copy, (1000, 1000))
        image_copy = np.roll(image_copy, 3 * random.randint(negative_roll, positive_roll))

        lower_blue = np.array([0, 50, 0])  # [R value, G value, B value]
        upper_blue = np.array([55, 255, 55])

        mask = cv2.inRange(image_copy, lower_blue, upper_blue)
        masked_image = np.copy(image_copy)
        masked_image[mask != 0] = [0, 0, 0]

        # loading new background image
        background_image = cv2.imread('backgrounds/brick.jpg')

        # Depending on the width of the background image change randint parameters
        background_image = np.roll(background_image, 3 * random.randint(-1800, 1800))

        # If the background image need to be resized uncomment this line.
        # background_image = cv2.resize(background_image, (1344, 1014))

        crop_background = background_image[0:1000, 0:1000]

        crop_background[mask == 0] = [0, 0, 0]

        final_image = crop_background + masked_image
        final_image = cv2.resize(final_image, (100, 100))


        #           FOLDER/       name-number.jpg
        cv2.imwrite(output_path + name + str(i) + '.jpg', final_image)
