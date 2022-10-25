import cv2 as cv
import numpy as np

base_images_address = 'images/'
base_results_address = 'results/'

sample = None


def resize_image(src_image, scale):
    width = int(src_image.shape[1] * scale / 100)
    height = int(src_image.shape[0] * scale / 100)

    dim = (width, height)

    return cv.resize(src_image, dim, interpolation=cv.INTER_AREA)


def mouse_handler(event, y, x, flags, param):
    global sample
    image = param['image']
    if event == cv.EVENT_LBUTTONDOWN:
        sample = [x, y]
        cv.circle(image, (y, x), 5, (0, 255, 0), 3)


def get_num_of_clusters(segments):
    labels_unique = np.unique(segments)
    return len(labels_unique)


def average_color(src_image, segments):
    labels_unique = np.unique(segments)

    for label in labels_unique:
        x, y = np.where(segments == label)
        src_image[x, y] = np.average(src_image[x, y])
