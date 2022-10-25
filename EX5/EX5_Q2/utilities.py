import cv2 as cv
import numpy as np

points = []

base_address_image = 'images/'
base_address_results = 'results/'


def get_gaussian_pyramid(src_image, num_levels):
    src_copy = src_image.copy()
    gaussian_pyramid = [src_copy]

    for index in range(num_levels):
        src_copy = cv.pyrDown(src_copy)
        gaussian_pyramid.append(src_copy)

    return gaussian_pyramid


def get_laplacian_pyramid(src_image, num_levels):
    gaussian_pyramid = get_gaussian_pyramid(src_image, num_levels)

    src_copy = gaussian_pyramid[num_levels]
    laplacian_pyramid = [src_copy]
    for index in range(num_levels, 0, -1):
        gaussian_expanded = cv.pyrUp(gaussian_pyramid[index])
        laplacian = cv.subtract(gaussian_pyramid[index - 1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)

    return laplacian_pyramid


def get_blended_pyramid(lp_source, lp_target, gp_mask):
    blended_pyramid = []

    for i in range(len(lp_source)):
        blend = gp_mask[i] * lp_source[i] + (1 - gp_mask[i]) * lp_target[i]
        blended_pyramid.append(blend)

    return blended_pyramid


def reconstruct_channel(blended_pyramid):
    reconstructed_channel = blended_pyramid[0]

    for i in range(1, 7):
        reconstructed_channel = cv.pyrUp(reconstructed_channel)
        reconstructed_channel = cv.add(blended_pyramid[i], reconstructed_channel)

    return reconstructed_channel


def blend_channel(source_channel, target_channel, mask):
    lp_source = get_laplacian_pyramid(source_channel, 6)
    lp_target = get_laplacian_pyramid(target_channel, 6)
    gp_mask = get_gaussian_pyramid(mask.copy(), 6)[::-1]

    blended_pyramid = get_blended_pyramid(lp_source, lp_target, gp_mask)
    blended_channel = reconstruct_channel(blended_pyramid)

    return blended_channel


def pyramid_blend(source, target, mask):
    result = np.zeros(target.shape)
    for channel in range(3):
        result[:, :, channel] = blend_channel(source[:, :, channel], target[:, :, channel], mask)

    return result


def mouse_handler(event, x, y, flags, param):
    apple_image = param['apple_image']
    if event == cv.EVENT_LBUTTONDOWN:
        point = np.array([x, y])
        points.append(point)
        cv.circle(apple_image, (x, y), 5, (0, 255, 0), 5)


def get_mask(shape, vertices):
    mask = cv.fillConvexPoly(np.zeros(shape, np.uint8), np.array([vertices]), 255)
    mask = (mask / 255).astype(np.float64)
    return mask
