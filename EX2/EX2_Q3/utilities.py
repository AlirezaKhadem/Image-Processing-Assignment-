import cv2 as cv
import numpy as np
from math import pow, sqrt, floor

base_images_address = "images/"
base_result_address = "results/"

image_width = 500


def warp_perspective(src_image, pts1, width=image_width):
    w = sqrt(pow(pts1[0][0] - pts1[1][0], 2) + pow(pts1[0][1] - pts1[1][1], 2))
    h = sqrt(pow(pts1[0][0] - pts1[2][0], 2) + pow(pts1[0][1] - pts1[2][1], 2))

    height = int(width * h // w)

    pts2 = np.float32([[0, 0], [0, width], [height, 0], [height, width]])
    m = cv.getPerspectiveTransform(pts1, pts2)
    m_inverse = np.linalg.inv(m)

    result = np.ndarray((height, width, 3))
    for i in range(height):
        for j in range(width):
            res = m_inverse @ [i, j, 1]
            res = res / res[2]

            x = floor(res[0])
            y = floor(res[1])

            a = res[0] - x
            b = res[1] - y

            for k in range(3):
                try:
                    l = np.array([1 - a, a])
                    p = np.array(
                        [[src_image[x, y, k], src_image[x, y + 1, k]],
                         [src_image[x + 1, y, k], src_image[x + 1, y + 1, k]]])
                    q = np.array([[1 - b], [b]])
                    r = l @ p @ q
                except:
                    continue
                result[i, j, k] = r

    return result

