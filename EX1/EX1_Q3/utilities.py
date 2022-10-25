import numpy as np
import cv2 as cv

image_base_file_address = "images/"
result_base_file_address = "results/"


def split_image_horizontally(input_image, t):
    h = input_image.shape[0] // t

    output = []
    for index in range(t):
        output.append(input_image[index * h: (index + 1) * h, :])

    return output


def edge_detection_filter(input_image):
    edge_detector_filter = np.array([0, -1, 0, -1, 0, 1, 0, 1, 0])

    output = cv.filter2D(input_image, -1, edge_detector_filter)
    return np.where(output > 15, 255, 0)


def ssd(input_image, template, k):
    input_image = input_image.astype(np.float_)
    template = template.astype(np.float_)

    eye = np.ones(template.shape)

    result = cv.filter2D(input_image ** 2, -1, eye.astype(np.float_)).astype(np.float_)
    result = result - 2 * cv.filter2D(input_image, -1, template)
    result = result - result.min() + 1.1

    for i in range(k):
        result = np.log(result)

    result = result * 255 / result.max()
    _minVal, _maxVal, min_location, max_location = cv.minMaxLoc(result, None)

    return min_location


def get_image_name():
    print("please enter name of image with its format")
    image_name = input()

    return image_name
