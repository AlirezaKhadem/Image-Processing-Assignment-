import cv2 as cv
import numpy as np
from math import sqrt
from skimage.segmentation import find_boundaries

base_images_address = 'images/'
base_results_address = 'results/'

indexes = [[-5, -5], [-5, -4], [-5, -3], [-5, -2], [-5, -1], [-5, 0], [-5, 1], [-5, 2], [-5, 3], [-5, 4], [-5, 5],
           [-4, -5], [-4, -4], [-4, -3], [-4, -2], [-4, -1], [-4, 0], [-4, 1], [-4, 2], [-4, 3], [-4, 4], [-4, 5],
           [-3, -5], [-3, -4], [-3, -3], [-3, -2], [-3, -1], [-3, 0], [-3, 1], [-3, 2], [-3, 3], [-3, 4], [-3, 5],
           [-2, -5], [-2, -4], [-2, -3], [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-2, 3], [-2, 4], [-2, 5],
           [-1, -5], [-1, -4], [-1, -3], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [-1, 3], [-1, 4], [-1, 5],
           [0, -5], [0, -4], [0, -3], [0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
           [1, -5], [1, -4], [1, -3], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
           [2, -5], [2, -4], [2, -3], [2, -2], [2, -1], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5],
           [3, -5], [3, -4], [3, -3], [3, -2], [3, -1], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
           [4, -5], [4, -4], [4, -3], [4, -2], [4, -1], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5],
           [5, -5], [5, -4], [5, -3], [5, -2], [5, -1], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]]


def resize_image(src_image, scale):
    width = int(src_image.shape[1] * scale / 100)
    height = int(src_image.shape[0] * scale / 100)

    return cv.resize(src_image, (width, height), interpolation=cv.INTER_AREA)


def apply_sobel_filter_x(gray_image):
    gray_image = gray_image.astype(np.float_)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filtered_image_x = cv.filter2D(gray_image, -1, sobel_x)

    return filtered_image_x


def apply_sobel_filter_y(gray_image):
    gray_image = gray_image.astype(np.float_)

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filtered_image_y = cv.filter2D(gray_image, -1, sobel_y)

    return filtered_image_y


def get_edges(src_image):
    gray_image = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)

    sobel_x = apply_sobel_filter_x(gray_image)
    sobel_y = apply_sobel_filter_y(gray_image)

    return sobel_x + sobel_y


def initialize_cluster_centers(scaled_image, number_of_clusters, src_image_edges, indexes):
    cluster_centers = list()

    image_shape = scaled_image.shape
    image_size = image_shape[0] * image_shape[1]
    S = int(sqrt(image_size / number_of_clusters))

    height = width = S / 2

    while height < image_shape[0]:
        while width < image_shape[1]:
            height, width = int(height), int(width)
            cluster_centers.append(
                [
                    height,
                    width,
                    (scaled_image[height, width][0], scaled_image[height, width][1], scaled_image[height, width][2]),
                    [(scaled_image[height, width][0], scaled_image[height, width][1], scaled_image[height, width][2])]
                ])
            width += S
        width = S / 2
        height += S

    perturb_cluster_centers(cluster_centers, src_image_edges, indexes)

    return cluster_centers, S


def perturb_cluster_centers(cluster_centers, src_image_edges, indexes):
    for cluster_center in cluster_centers:
        minimum = np.inf

        for index in indexes:
            x = cluster_center[0] + index[0]
            y = cluster_center[1] + index[1]
            if x < src_image_edges.shape[0] and y < src_image_edges.shape[1]:
                gradient = src_image_edges[x, y]
                if gradient < minimum:
                    minimum = gradient
                    cluster_center = [cluster_center[0] + index[0], cluster_center[1] + index[1]]


def cal_dist(cluster_center, pixel_x, pixel_y, scaled_image, S):
    cluster_center_x = cluster_center[0]
    cluster_center_y = cluster_center[1]
    cluster_center_l = cluster_center[2][0]
    cluster_center_a = cluster_center[2][1]
    cluster_center_b = cluster_center[2][2]

    pixel_l = scaled_image[pixel_x, pixel_y][0]
    pixel_a = scaled_image[pixel_x, pixel_y][1]
    pixel_b = scaled_image[pixel_x, pixel_y][2]

    d_lab = pow(cluster_center_l - pixel_l, 2) + pow(cluster_center_a - pixel_a, 2) + pow(
        cluster_center_b - pixel_b, 2)

    d_xy = pow(float(cluster_center_x) - pixel_x, 2) + pow(float(cluster_center_y) - pixel_y, 2)

    return d_lab + 15 / S * d_xy


def update_cluster_center(cluster_centers, scaled_image, k, x, y):
    cluster_center = cluster_centers[k]
    average = np.average(scaled_image[x, y], axis=0)
    cluster_center[3] = average


def slic(cluster_centers, scaled_image, S, f, data, segments=None):
    if segments is None:
        segments = np.zeros(scaled_image.shape[:2], dtype=np.int)

    for k in range(len(cluster_centers)):
        cluster_center = cluster_centers[k]

        temp_color = scaled_image[
                     max(cluster_center[0] - S, 0): min(cluster_center[0] + S, scaled_image.shape[0] - 1),
                     max(cluster_center[1] - S, 0): min(cluster_center[1] + S, scaled_image.shape[1] - 1)]
        temp_data = data[
                    max(cluster_center[0] - S, 0): min(cluster_center[0] + S, scaled_image.shape[0] - 1),
                    max(cluster_center[1] - S, 0): min(cluster_center[1] + S, scaled_image.shape[1] - 1), 1]

        cluster_center_color = np.ones((temp_color.shape[0], temp_color.shape[1], 3)) * cluster_center[3]

        temp_color = temp_color - cluster_center_color
        temp_color = temp_color ** 2
        temp_color = np.sum(temp_color, axis=2)
        temp_color = np.sqrt(temp_color)

        temp_coordinate_x = f[-1 * temp_color.shape[0]:, -1 * temp_color.shape[1]:, 0]
        temp_coordinate_y = f[-1 * temp_color.shape[0]:, -1 * temp_color.shape[1]:, 1]

        temp_coordinate_x = temp_coordinate_x ** 2
        temp_coordinate_y = temp_coordinate_y ** 2

        temp_coordinate = temp_coordinate_x + temp_coordinate_y
        temp_coordinate = np.sqrt(temp_coordinate)

        dist = temp_color + 22 / S * temp_coordinate

        x, y = np.where(dist < temp_data)

        segments[x + cluster_center[0] - S, y + cluster_center[1] - S] = k + 1
        data[x + cluster_center[0] - S, y + cluster_center[1] - S, 0] = k + 1
        data[x + cluster_center[0] - S, y + cluster_center[1] - S, 1] = dist[x, y]

        update_cluster_center(cluster_centers,
                              scaled_image, k,
                              x + cluster_center[0] - S,
                              y + cluster_center[1] - S)

    return segments


def draw_boundaries(segments, scaled_image):
    boundary = find_boundaries(segments).astype(np.uint8)

    x, y = np.where(boundary == 1)
    scaled_image[x, y] = 255

    x, y = np.where(segments == 0)
    scaled_image[x, y] = [0, 0, 0]
    print(len(np.unique(segments)))


def get_inputs():
    print('please enter number Of clusters: ')
    number_of_clusters = int(input())
    print('please enter output image scale [0 100]: ')
    scale = float(input())

    return number_of_clusters, scale
