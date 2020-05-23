# −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−------------------#
#  Author: Sander Hellesø and William Svea-Lochert
#  Run script:  python capture.py <datapath> <outputpath> <imgsize>
#
# This script is for creating a dataset. You feed it folder containing
# videos, and it will return as many  * pictures for every sec as the
# FPS of the video file has, for every video found in the passed in folder.
#
# −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−------------------#

# Importing all necessary libraries
import cv2
import os
import sys
import numpy as np
import random
import pickle
from datetime import datetime

videosSourcePath = sys.argv[1]
directoryName = sys.argv[2]
imgSize = int(sys.argv[3])

CLASS_CATEGORIES = ["forward", "right", "left", "backward", "stop"]
trainingData = []


# recurse over dirs, and extract frames from every video found
def read_dir_and_start_frame_extraction(path):
    directory = os.fsencode(path)

    print("Extracting frames from files found under: " + path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        fullFilePath = create_path(path, filename)

        if filename.endswith(".h264") or filename.endswith(".mp4"):
            extract_frames_from_video(fullFilePath)
        elif os.path.isdir(fullFilePath):
            read_dir_and_start_frame_extraction(fullFilePath)


def extract_frames_from_video(videoSource):
    # Read the video from specified source
    cam = cv2.VideoCapture(videoSource)

    category = videoSource.split('-')[1]
    classNum = CLASS_CATEGORIES.index(category)
    imageOutputPath = build_output_image_path(category)
    currentframe = 0

    while True:

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            videoFileName = clean_filename(os.path.basename(videoSource))
            outputImageName = videoFileName + '-frame-' + str(currentframe) + '-bW.jpg'
            outputDestination = create_path(imageOutputPath, outputImageName)

            # writing the extracted images
            cv2.imwrite(outputDestination, frame)

            # manipulate extracted image
            img_ = cv2.imread(outputDestination, cv2.IMREAD_ANYCOLOR)
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            img_ = cv2.resize(gray, (imgSize, imgSize))

            # overwrite extracted image with updated properties
            cv2.imwrite(outputDestination, img=img_)

            # create image array from image and add to training data
            add_img_array_to_training_data(outputDestination, classNum)

            # increasing counter so that it will
            # correctly name the file with current frame counter
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


# helper to clean the filename from its extension
def clean_filename(fileName):
    strList = fileName.split('.')
    strList.pop()

    return "".join(strList)


# helper to create joined path with OS specific delimiters
def create_path(root, path):
    return os.path.join(root, path)


# combine passed in directory name and source category
def build_output_image_path(category):
    imageOutputPath = create_path(directoryName, category)

    try:
        # creating a folder named data
        if not os.path.exists(imageOutputPath):
            os.makedirs(imageOutputPath)

        # if not created, then raise error
    except OSError:
        print('Error: Creating directory: ' + imageOutputPath)

    return imageOutputPath


def add_img_array_to_training_data(imgPath, classNum):
    imgArray = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    trainingData.append([imgArray, classNum])


def create_training_data():
    random.shuffle(trainingData)

    # features
    X_PICKLE_OUT_PATH = create_path(directoryName, "X.pickle")
    X = []

    # labels
    Y_PICKLE_OUT_PATH = create_path(directoryName, "y.pickle")
    y = []

    for features, label in trainingData:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, imgSize, imgSize, 1)

    create_pickle(X, X_PICKLE_OUT_PATH)
    create_pickle(y, Y_PICKLE_OUT_PATH)


def create_pickle(pickleData, path):
    pickleOut = open(path, "wb")
    pickle.dump(pickleData, pickleOut)
    pickleOut.close()


def start():
    startTime = datetime.now()

    read_dir_and_start_frame_extraction(videosSourcePath)
    create_training_data()

    timeToRun = datetime.now() - startTime
    print("Done in " + str(timeToRun.seconds) + " seconds.")


start()