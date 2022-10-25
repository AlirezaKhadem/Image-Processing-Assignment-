import cv2 as cv
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve

base_address_images = 'images/'
base_address_results = 'results/'

delta_x = 2500
delta_y = 150


def get_poisson_matrix(n, m):
    D = scipy.sparse.lil_matrix((m, m))
    D.setdiag(-1, -1)
    D.setdiag(4)
    D.setdiag(-1, 1)

    A = scipy.sparse.block_diag([D] * n).tolil()

    A.setdiag(-1, 1 * m)
    A.setdiag(-1, -1 * m)

    return A


def warp_source_image(source_image, delta_x, delta_y, shape):
    translation = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    warped_source_image = cv.warpAffine(source_image, translation, (shape[1], shape[0]))

    return warped_source_image


def get_mat_b(A, source_flat, target_flat, mask_flat):
    b = A.dot(source_flat)
    b[mask_flat == 0] = target_flat[mask_flat == 0]

    return b


def get_mat_A(mask, shape):
    A = get_poisson_matrix(shape[0], shape[1])

    for row in range(1, shape[0] - 1):
        for col in range(1, shape[1] - 1):
            if mask[row, col] == 0:
                index = col + row * shape[1]
                A[index, index] = 1
                A[index, index + 1] = 0
                A[index, index - 1] = 0
                A[index, index + shape[1]] = 0
                A[index, index - shape[1]] = 0

    A = A.tocsc()

    return A


def scale(image, min, max):
    image[image <= min] = min
    image[image >= max] = max
    image = image.astype(np.uint8)

    return image


def blend_channel(A, source, target, mask, channel, shape):
    source_flat = source[:, :, channel].flatten()
    target_flat = target[:, :, channel].flatten()
    mask_flat = mask.flatten()

    b = get_mat_b(A, source_flat, target_flat, mask_flat)

    f = spsolve(A, b)
    f = f.reshape((shape[0], shape[1]))
    f = scale(f, 0, 255)

    return f


def blend(warped_source, target, mask):
    source = warped_source
    shape = target.shape

    A = get_mat_A(mask, shape)

    for channel in range(source.shape[2]):
        blended_channel = blend_channel(A, source, target, mask, channel, shape)

        target[:, :, channel] = blended_channel

    return target
