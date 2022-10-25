import cv2 as cv
import numpy as np

from copy import deepcopy

base_images_address = 'images/'
base_results_address = 'results/'


def apply_unsharp_filter(input_channel):
    output_image = deepcopy(input_channel)

    filter = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1])
    output_image = cv.filter2D(output_image, -1, filter)

    return output_image


def unsharp(src_image):
    b, g, r = cv.split(src_image)

    unsharp_b = apply_unsharp_filter(b)
    unsharp_g = apply_unsharp_filter(g)
    unsharp_r = apply_unsharp_filter(r)

    return cv.merge((unsharp_b, unsharp_g, unsharp_r))


def sharpe(src_image, alpha):
    src_image = src_image.astype(np.float_)
    unsharp_image = unsharp(src_image)

    output_image = src_image + alpha * unsharp_image
    return output_image
