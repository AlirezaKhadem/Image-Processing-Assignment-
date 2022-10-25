import cv2 as cv
import numpy as np


def match(near_image, far_image):

    pts1 = np.float32([[259, 256], [264, 598], [64, 432]])
    pts2 = np.float32([[254, 253], [265, 599], [64, 430]])

    transform = cv.getAffineTransform(pts1, pts2)
    warped = cv.warpAffine(near_image, transform, (far_image.shape[1], far_image.shape[0]))

    return warped
