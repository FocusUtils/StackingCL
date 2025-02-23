import cv2
import numpy as np
import math

## white to black gradient
gray = np.zeros((256, 256), dtype=np.uint8)
for i in range(256):
    gray[:, i] = i


def get_colortone(t):
    #       B                   G                   R
    return [255 * (1 - t),      80 * (t),    255 * t]


BLUE2ORANGE_LUT = np.zeros((256, 1, 3), dtype=np.uint8)
ORANGE2BLUE_LUT = np.zeros((256, 1, 3), dtype=np.uint8)
for i in range(256):
    
    
    t = i / 255.0  # Normalize
    t = 0.2 * math.tan(2.3 * (t - 0.5)) + 0.5
    BLUE2ORANGE_LUT[i, 0] = get_colortone(t)
    ORANGE2BLUE_LUT[i, 0] = get_colortone(1 - t)

print(BLUE2ORANGE_LUT)
# Apply the color mapping
colorized = cv2.LUT(cv2.merge([gray, gray, gray]), BLUE2ORANGE_LUT)

cv2.imshow('gray', gray)
cv2.imshow('colorized', colorized)
cv2.waitKey(0)
