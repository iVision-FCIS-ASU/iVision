import cv2
import numpy as np

def get_canny(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)

    high = np.percentile(mag, 90)
    low = 0.5 * high

    blur = cv2.GaussianBlur(image, (5, 5), 1.4)
    edges = cv2.Canny(blur, low, high)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return edges
