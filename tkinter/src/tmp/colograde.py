import cv2
import numpy as np
import math

## white to black gradient
gray = np.zeros((256, 256), dtype=np.uint8)
for i in range(256):
    gray[:, i] = i


lut = np.zeros((256, 1, 3), dtype=np.uint8)
for i in range(256):
    
    
    t = i / 255.0  # Normalize
    t = 0.2 * math.tan(2.3 * (t - 0.5)) + 0.5
    lut[i, 0] = [255 * (1 - t), 100 * (1 - t/3), 255 * t]  # [B, G, R]

print(lut)
# Apply the color mapping
colorized = cv2.LUT(cv2.merge([gray, gray, gray]), lut)

cv2.imshow('gray', gray)
cv2.imshow('colorized', colorized)
cv2.waitKey(0)
