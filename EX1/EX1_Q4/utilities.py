import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image_base_file_address = "images/"
result_base_file_address = "results/"


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def histogram(input_image):
    return np.histogram(input_image.flatten(), 256, [0, 255])[0]


def one_channel_histogram_specified(input_channel, target_channel):
    output = input_channel

    target_histogram = histogram(target_channel)
    input_histogram = histogram(input_channel)

    cdf_target = target_histogram.cumsum()
    cdf_input = input_histogram.cumsum()

    normalized_cdf_target = cdf_target / cdf_target.max()
    normalized_cdf_input = cdf_input / cdf_input.max()

    for index in range(256)[::-1]:
        value = find_nearest(normalized_cdf_target, normalized_cdf_input[index])
        output[output == index] = value

    return output


def histogram_specified(input_image, target_image):
    # tb : Abbreviation for target blue
    # tg : Abbreviation for target green
    # tr : Abbreviation for target red
    b, g, r = cv.split(input_image)
    tb, tg, tr = cv.split(target_image)

    blue_histogram_specified = one_channel_histogram_specified(b, tb)
    green_histogram_specified = one_channel_histogram_specified(g, tg)
    red_histogram_specified = one_channel_histogram_specified(r, tr)

    output_image = cv.merge((blue_histogram_specified,
                             green_histogram_specified,
                             red_histogram_specified))

    return output_image
