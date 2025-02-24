import cv2
import numpy as np

def convert_gray_arr_to_gray_image(arr, width, height):
    return cv2.flip(cv2.rotate(arr.reshape(width, height), cv2.ROTATE_90_CLOCKWISE), 1)

def convert_gray_arr_to_image(arr, width, height):
    ## check if its uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return cv2.cvtColor(convert_gray_arr_to_gray_image(arr, width, height), cv2.COLOR_GRAY2RGB)


def convert_color_arr_to_image(arr, width, height):
    return cv2.cvtColor(arr.reshape(height, width, 3), cv2.COLOR_BGR2RGB)