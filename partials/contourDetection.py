## create a script to call the open file dialog for an image and then apply the cv2 contour detection to the image
## and display the image with the contours

import cv2
import numpy as np
from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt
from PIL import Image

def openImage():
    Tk().withdraw()
    file_path = filedialog.askopenfilename()
    image = cv2.imread(file_path, 0)
    return image

def detectContours(image, threshold):
    # threshold the image
    ret, thresh = cv2.threshold(image, threshold, 255, 0)
    # find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def drawContours(image, contours):
    # draw the contours
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    return image

def updateThreshold(val):
    global image, window_name
    threshold = cv2.getTrackbarPos('Threshold', window_name)
    contours = detectContours(image, threshold)
    contoured_image = drawContours(image.copy(), contours)
    ## retain aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]
    max_height = 900
    max_width = 1920
    height = min(max_height, int(max_width / aspect_ratio))
    width = min(max_width, int(max_height * aspect_ratio))
    contoured_image = cv2.resize(contoured_image, (width, height))
    cv2.imshow(window_name, contoured_image)

if __name__ == "__main__":
    image = openImage()
    window_name = 'Contour Detection'
    cv2.namedWindow(window_name)
    cv2.createTrackbar('Threshold', window_name, 127, 255, updateThreshold)
    updateThreshold(127)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
