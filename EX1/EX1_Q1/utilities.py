import numpy as np
import cv2 as cv
from math import log

base_results_address = 'results/'
base_images_address = 'images/'


def gamma_transformation(src_image, gamma):
    return 255 * np.power(src_image / 255, gamma)


def log_transformation(src_image, coefficient):
    return np.log(src_image * coefficient + 1) * 255 / log(256 * coefficient)


def get_histogram(src_image):
    r, g, b = cv.split(src_image)

    b_pdf = np.histogram(b.flatten(), 256, [0, 255])[0]
    g_pdf = np.histogram(g.flatten(), 256, [0, 255])[0]
    r_pdf = np.histogram(r.flatten(), 256, [0, 255])[0]

    return b_pdf, g_pdf, r_pdf


def get_cdf(src_image):
    b_pdf, g_pdf, r_pdf = get_histogram(src_image)

    b_cdf = b_pdf.cumsum()
    g_cdf = g_pdf.cumsum()
    r_cdf = r_pdf.cumsum()

    return b_cdf, g_cdf, r_cdf


def plot_cdf(ax, index, b_cdf, g_cdf, r_cdf):
    ax[index, 0].plot(b_cdf / b_cdf.max())
    ax[index, 1].plot(g_cdf / g_cdf.max())
    ax[index, 2].plot(r_cdf / r_cdf.max())


def plot_pdf(ax, index, b_pdf, g_pdf, r_pdf):
    ax[index, 0].plot(b_pdf / b_pdf.max())
    ax[index, 1].plot(g_pdf / g_pdf.max())
    ax[index, 2].plot(r_pdf / r_pdf.max())
