import cv2 as cv
import numpy as np

base_images_address = "images/"
base_results_address = "results/"
number_of_object = 9


def canny(src_image):
    high_thresh, thresh_im = cv.threshold(src_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    low_thresh = low_thresh + 200

    return cv.Canny(src_image, low_thresh, high_thresh)


def cut_object_of_patch(gray_template_patch, template_patch):
    edge_detected_template = canny(gray_template_patch)

    active_px = np.argwhere(edge_detected_template != 0)
    active_px = active_px[:, [1, 0]]
    x, y, w, h = cv.boundingRect(active_px)

    return template_patch[y:y + h, x:x + w]


def normalized_cross_correlation(src_image, input_template):
    input_template = input_template - np.average(input_template)
    src_image = src_image - np.average(src_image)

    x = np.sqrt((input_template ** 2).sum() + (src_image ** 2).sum())

    result = cv.filter2D(src_image, -1, input_template)
    result = result / x

    result = result - result.min()
    result = result / result.max() * 255

    return result


def draw_rectangle_around_detected_template(src_image, result, w, h, k):
    for i in range(k):
        _minVal, _maxVal, min_loc, max_loc = cv.minMaxLoc(result, None)
        y, x = max_loc

        result[x - w // 2 - 150: x + w // 2 + 150, y - h // 2: y + h // 2] = result.min()

        cv.rectangle(src_image, (y - h // 2, x - w // 2), (y + h // 2, x + w // 2), (0, 0, 255), 5)


def apply_gamma_transformation(src_image, gamma):
    return 255 * np.power(src_image / 255, gamma)
