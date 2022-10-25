import cv2 as cv
import numpy as np
from math import sqrt

base_images_address = 'images/'
base_results_address = 'results/'

vertices_ = list()
centers = list()
iteration_counter = 0


def add_remove_vertex(vertices, add_threshold, remove_threshold):
    vertices_array_length = len(vertices)

    for index in range(len(vertices))[::-1]:
        if index != vertices_array_length - 1:
            sub_x = vertices[index + 1][0] - vertices[index][0]
            sub_y = vertices[index + 1][1] - vertices[index][1]

            d2 = pow(sub_x, 2) + pow(sub_y, 2)
            if sqrt(d2) > add_threshold:
                new_vertex_x = (vertices[index + 1][0] + vertices[index][0]) // 2
                new_vertex_y = (vertices[index + 1][1] + vertices[index][1]) // 2

                vertices.insert(index + 1, [new_vertex_x, new_vertex_y])
            if sqrt(d2) < remove_threshold:
                vertices.pop(index + 1)
        else:
            # pass
            sub_x = vertices[index][0] - vertices[0][0]
            sub_y = vertices[index][1] - vertices[0][1]

            d2 = pow(sub_x, 2) + pow(sub_y, 2)

            if sqrt(d2) > add_threshold:
                new_vertex_x = (vertices[index][0] + vertices[0][0]) // 2
                new_vertex_y = (vertices[index][1] + vertices[0][1]) // 2

                vertices.append([new_vertex_x, new_vertex_y])


def cal_internal_energy(vertices, index):
    vertices_array_length = len(vertices)

    if index != vertices_array_length - 1:
        sub_x = vertices[index + 1][0] - vertices[index][0]
        sub_y = vertices[index + 1][1] - vertices[index][1]

        d2 = pow(sub_x, 2) + pow(sub_y, 2)

        return d2
    else:
        sub_x = vertices[index][0] - vertices[0][0]
        sub_y = vertices[index][1] - vertices[0][1]

        d2 = pow(sub_x, 2) + pow(sub_y, 2)

        return d2


def cal_average_length_of_contoure(vertices):
    data = np.ndarray(len(vertices))
    for index in range(len(vertices)):
        if index != len(vertices) - 1:
            sub_x = vertices[index + 1][0] - vertices[index][0]
            sub_y = vertices[index + 1][1] - vertices[index][1]

            data[index] = pow(sub_x, 2) + pow(sub_y, 2)
        else:
            sub_x = vertices[index][0] - vertices[0][0]
            sub_y = vertices[index][1] - vertices[0][1]

            data[index] = pow(sub_x, 2) + pow(sub_y, 2)

    return np.average(data)


def cal_external_energy(edge_detected_image, vertices, index):
    return edge_detected_image[vertices[index][1], vertices[index][0]]


def cal_centers_energy(vertices, index, centers):
    if len(centers) != 0:
        summation = 0
        for center in centers:
            sub_x = vertices[index][0] - center[0]
            sub_y = vertices[index][1] - center[1]
            summation += pow(sub_x, 2) + pow(sub_y, 2)

        return summation / len(centers)
    else:
        global vertices_

        M = cv.moments(np.array(vertices_))

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        sub_x = vertices[index][0] - cX
        sub_y = vertices[index][1] - cY

        return pow(sub_x, 2) + pow(sub_y, 2)


def cal_energy(edge_detected_image, vertices, alpha, gamma, beta, average_of_length, center):
    summation = 0

    for index in range(len(vertices)):
        external_energy = cal_external_energy(edge_detected_image, vertices, index)
        internal_energy = cal_internal_energy(vertices, index)
        centers_energy = cal_centers_energy(vertices, index, center)

        summation += -1 * gamma * external_energy + alpha * pow(internal_energy - 0.6 * average_of_length,
                                                                2) + beta * centers_energy

    return summation


def iteration(edge_detected_image, vertices, indexes, centers, alpha, gamma, beta, k_window, max_iteration):
    global iteration_counter
    while iteration_counter < max_iteration:
        halt = True

        average_of_length_ = cal_average_length_of_contoure(vertices)

        data = np.ndarray((k_window * k_window, len(vertices), 2))

        for k in range(len(vertices)):
            for l in range(k_window * k_window):
                if k == 0:
                    data[l][k] = [l, 0]
                    continue

                minimum = np.inf

                for i in (np.array(range(k_window)) - k_window // 2):

                    for j in (np.array(range(k_window)) - k_window // 2):
                        total_vertex = [vertices[k][0] + indexes[l][0], vertices[k][1] + indexes[l][1]]
                        pre_vertex = [vertices[k - 1][0] + i, vertices[k - 1][1] + j]

                        index = k_window * (k_window // 2 + i) + j + k_window // 2

                        energy = cal_energy(edge_detected_image,
                                            [total_vertex, pre_vertex],
                                            alpha, gamma, beta,
                                            average_of_length_, centers) + data[index][k - 1][1]

                        if energy < minimum:
                            minimum = energy
                            data[l][k] = [index, energy]

        arg_min = np.argmin(data[:, len(vertices) - 1, 1])
        while_counter = len(vertices) - 1
        next_vertex = int(data[arg_min, while_counter, 0])

        while while_counter != -1:

            if while_counter == 0:
                vertices[while_counter] = vertices[len(vertices) - 1]

            if indexes[next_vertex][0] != 0 or indexes[next_vertex][1] != 0:
                halt = False

            vertices[while_counter][0] = vertices[while_counter][0] + indexes[next_vertex][0]
            vertices[while_counter][1] = vertices[while_counter][1] + indexes[next_vertex][1]

            next_vertex = int(data[next_vertex, while_counter, 0])

            while_counter -= 1

        add_remove_vertex(vertices, 20, 5)
        iteration_counter += 1

        if halt:
            break

    temp_image = cv.imread(base_images_address + 'tasbih.jpg')

    cv.drawContours(temp_image, [np.array(vertices)], -1, (0, 255, 0), 2)
    for vertex in vertices:
        cv.circle(temp_image, (vertex[0], vertex[1]), 2, (0, 0, 100), 1)

    cv.imwrite(base_results_address + "res09.jpg", temp_image)


def apply_sobel_filter_x(gray_image):
    gray_image = gray_image.astype(np.float_)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filtered_image_x = cv.filter2D(gray_image, -1, sobel_x)

    return filtered_image_x ** 2


def apply_sobel_filter_y(gray_image):
    gray_image = gray_image.astype(np.float_)

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filtered_image_y = cv.filter2D(gray_image, -1, sobel_y)

    return filtered_image_y ** 2


def get_gradient(gray_image):
    x = apply_sobel_filter_x(gray_image)
    y = apply_sobel_filter_y(gray_image)

    gradient = x + y

    return gradient
