import numpy as np
import cv2 as cv
from random import randint

images_address_base = 'images/'
results_address_base = 'results/'


def initialized_first_block(src_image, target_image, height_of_block, width_of_block):
    x = randint(0, src_image.shape[0] - height_of_block)
    y = randint(0, src_image.shape[1] - width_of_block)

    block = src_image[x: x + height_of_block, y: y + width_of_block]
    target_image[0: height_of_block, 0: width_of_block] = block

    return target_image


def find_matching(src_image, patch, height_of_block, width_of_block):
    shape = src_image.shape
    matching = cv.matchTemplate(src_image[:shape[0] - height_of_block, :shape[1] - width_of_block],
                                patch.astype(np.uint8), cv.TM_CCORR_NORMED)
    locations = np.where(matching >= matching.max() - .0001)
    
    random_num = randint(0, len(locations[0]) - 1)
    point = [locations[0][max(random_num, 0)], locations[1][max(random_num, 0)]]

    return src_image[point[0]: point[0] + height_of_block, point[1]:point[1] + width_of_block, :]


def find_l_matching(src_image, patch, height_of_block, width_of_block, overlap):
    mask = np.ones((height_of_block, width_of_block, 3))
    mask[overlap:, overlap:] = 0

    matching = cv.matchTemplate(src_image, patch.astype(np.uint8), cv.TM_CCORR_NORMED, mask=mask)

    locations = np.where(matching >= matching.max() - .0001)

    random_num = randint(0, len(locations[0]))
    point = [locations[0][max(random_num - 1, 0)], locations[1][max(random_num - 1, 0)]]

    return src_image[point[0]: point[0] + height_of_block, point[1]:point[1] + width_of_block, :]


def vertically_min_cut(first_patch, second_patch):
    difference_of_squares = (cv.cvtColor(first_patch.astype(np.uint8), cv.COLOR_BGR2GRAY) -
                             cv.cvtColor(second_patch.astype(np.uint8), cv.COLOR_BGR2GRAY)) ** 2

    shape = first_patch.shape
    data = np.full((shape[0], shape[1], 2), np.inf)

    for row in range(shape[0]):
        for column in range(shape[1]):
            if row == 0:
                data[row, column] = [column, difference_of_squares[row, column]]
                continue

            left_up_column = max(0, column - 1)
            up_column = column
            right_up_column = min(shape[1] - 1, column + 1)

            left_up_cost = data[row - 1, left_up_column, 1]
            up_cost = data[row - 1, up_column, 1]
            right_up_cost = data[row - 1, right_up_column, 1]

            if left_up_cost < up_cost and left_up_cost < right_up_cost:
                data[row, column] = [left_up_column,
                                     data[row - 1, left_up_column, 1] + difference_of_squares[row, column]]
            elif right_up_cost < up_cost and right_up_cost < left_up_cost:
                data[row, column] = [right_up_column,
                                     data[row - 1, right_up_column, 1] + difference_of_squares[row, column]]
            else:
                data[row, column] = [up_column, data[row - 1, up_column, 1] + difference_of_squares[row, column]]

    arg_min = np.argmin(data[shape[0] - 1:, :, 1])
    while_counter = shape[0] - 1
    result = np.zeros(shape)

    while while_counter != -1:
        result[while_counter, 0:arg_min] = first_patch[while_counter, 0:arg_min]
        result[while_counter, arg_min:] = second_patch[while_counter, arg_min:]

        arg_min = int(data[while_counter, arg_min, 0])

        while_counter -= 1

    return result


def horizontally_min_cut(first_patch, second_patch):
    difference_of_squares = (cv.cvtColor(first_patch.astype(np.uint8), cv.COLOR_BGR2GRAY) -
                             cv.cvtColor(second_patch.astype(np.uint8), cv.COLOR_BGR2GRAY)) ** 2

    shape = first_patch.shape
    data = np.full((shape[0], shape[1], 2), np.inf)

    for column in range(shape[1]):
        for row in range(shape[0]):
            if column == 0:
                data[row, column] = [row, difference_of_squares[row, column]]
                continue

            left_up_row = max(0, row - 1)
            left_row = row
            left_down_row = min(shape[0] - 1, row + 1)

            left_up_cost = data[left_up_row, column - 1, 1]
            left_cost = data[left_row, column - 1, 1]
            left_down_cost = data[left_down_row, column - 1, 1]

            if left_up_cost < left_cost and left_up_cost < left_down_cost:
                data[row, column] = [left_up_row, left_up_cost + difference_of_squares[row, column]]
            elif left_down_cost < left_cost and left_down_cost < left_up_cost:
                data[row, column] = [left_down_row, left_down_cost + difference_of_squares[row, column]]
            else:
                data[row, column] = [left_row, left_cost + difference_of_squares[row, column]]

    arg_min = np.argmin(data[:, shape[1] - 1:, 1])
    while_counter = shape[1] - 1
    result = np.zeros(first_patch.shape)

    while while_counter != -1:
        result[0:arg_min, while_counter] = first_patch[0:arg_min, while_counter]
        result[arg_min:, while_counter] = second_patch[arg_min:, while_counter]

        arg_min = int(data[arg_min, while_counter, 0])

        while_counter -= 1

    return result


def vertically_min_cut2(first_patch, second_patch):
    difference_of_squares = (cv.cvtColor(first_patch.astype(np.uint8), cv.COLOR_BGR2GRAY) -
                             cv.cvtColor(second_patch.astype(np.uint8), cv.COLOR_BGR2GRAY)) ** 2

    shape = first_patch.shape
    data = np.full((shape[0], shape[1], 2), np.inf)

    for row in range(shape[0]):
        for column in range(shape[1]):
            if row == 0:
                data[row, column] = [column, difference_of_squares[row, column]]
                continue

            left_up_column = max(0, column - 1)
            up_column = column
            right_up_column = min(shape[1] - 1, column + 1)

            left_up_cost = data[row - 1, left_up_column, 1]
            up_cost = data[row - 1, up_column, 1]
            right_up_cost = data[row - 1, right_up_column, 1]

            if left_up_cost < up_cost and left_up_cost < right_up_cost:
                data[row, column] = [left_up_column,
                                     data[row - 1, left_up_column, 1] + difference_of_squares[row, column]]
            elif right_up_cost < up_cost and right_up_cost < left_up_cost:
                data[row, column] = [right_up_column,
                                     data[row - 1, right_up_column, 1] + difference_of_squares[row, column]]
            else:
                data[row, column] = [up_column, data[row - 1, up_column, 1] + difference_of_squares[row, column]]

    arg_min = np.argmin(data[shape[0] - 1:, :, 1])
    while_counter = shape[0] - 1
    result = np.zeros(shape)
    cut = np.zeros(shape)

    while while_counter != -1:
        result[while_counter, 0:arg_min] = first_patch[while_counter, 0:arg_min]
        cut[while_counter, arg_min] = 1

        arg_min = int(data[while_counter, arg_min, 0])

        while_counter -= 1

    return result, cut


def horizontally_min_cut2(first_patch, second_patch):
    difference_of_squares = (cv.cvtColor(first_patch.astype(np.uint8), cv.COLOR_BGR2GRAY) -
                             cv.cvtColor(second_patch.astype(np.uint8), cv.COLOR_BGR2GRAY)) ** 2

    shape = first_patch.shape
    data = np.full((shape[0], shape[1], 2), np.inf)

    for column in range(shape[1]):
        for row in range(shape[0]):
            if column == 0:
                data[row, column] = [row, difference_of_squares[row, column]]
                continue

            left_up_row = max(0, row - 1)
            left_row = row
            left_down_row = min(shape[0] - 1, row + 1)

            left_up_cost = data[left_up_row, column - 1, 1]
            left_cost = data[left_row, column - 1, 1]
            left_down_cost = data[left_down_row, column - 1, 1]

            if left_up_cost < left_cost and left_up_cost < left_down_cost:
                data[row, column] = [left_up_row, left_up_cost + difference_of_squares[row, column]]
            elif left_down_cost < left_cost and left_down_cost < left_up_cost:
                data[row, column] = [left_down_row, left_down_cost + difference_of_squares[row, column]]
            else:
                data[row, column] = [left_row, left_cost + difference_of_squares[row, column]]

    arg_min = np.argmin(data[:, shape[1] - 1:, 1])
    while_counter = shape[1] - 1
    result = np.zeros(first_patch.shape)
    cut = np.full(first_patch.shape, 0)

    while while_counter != -1:
        result[0:arg_min, while_counter] = first_patch[0:arg_min, while_counter]
        cut[arg_min, while_counter] = 1

        arg_min = int(data[arg_min, while_counter, 0])

        while_counter -= 1

    return result, cut


def texture_synthesis(src_image, target_image, height_of_block, width_of_block, overlap):
    initialized_first_block(src_image, target_image, height_of_block, width_of_block)

    number_of_blocks_in_col = (target_image.shape[1] - width_of_block) // (width_of_block - overlap)
    number_of_blocks_in_row = (target_image.shape[0] - height_of_block) // (height_of_block - overlap)

    for column in range(number_of_blocks_in_col):
        for row in range(number_of_blocks_in_row):
            if row == 0:
                y_1 = (column + 1) * (width_of_block - overlap)
                x_1 = row * (height_of_block - overlap)

                first_patch = target_image[x_1: x_1 + height_of_block, y_1:y_1 + overlap, :]
                second_patch = find_matching(src_image, first_patch, height_of_block, width_of_block)
                res = vertically_min_cut(first_patch, second_patch[:, 0:overlap, :])

                target_image[x_1: x_1 + height_of_block, y_1:y_1 + overlap, :] = cv.boxFilter(res, -1, (3, 3))
                target_image[x_1: x_1 + height_of_block, y_1 + overlap: y_1 + width_of_block, :] = second_patch[
                                                                                                   :,
                                                                                                   overlap:,
                                                                                                   :]
            else:

                y_1 = column * (width_of_block - overlap)
                x_1 = row * (height_of_block - overlap)

                if column == 0:
                    first_patch = target_image[x_1: x_1 + overlap, y_1: y_1 + width_of_block, :]
                    second_patch = find_matching(src_image, first_patch, height_of_block, width_of_block)

                    res = horizontally_min_cut(first_patch, second_patch[:overlap, :, :])

                    target_image[x_1:x_1 + overlap, y_1:y_1 + width_of_block, :] = cv.boxFilter(res, -1, (3, 3))
                    target_image[x_1 + overlap:x_1 + height_of_block, y_1:y_1 + width_of_block, :] = second_patch[
                                                                                                     overlap:, :, :]
                else:

                    first_patch = target_image[x_1: x_1 + height_of_block, y_1: y_1 + width_of_block, :]
                    second_patch = find_l_matching(src_image, first_patch, height_of_block, width_of_block, overlap)

                    v_res, v_res2 = vertically_min_cut2(first_patch[:, :overlap, :], second_patch[:, :overlap, :])
                    h_res, h_res2 = horizontally_min_cut2(first_patch[:overlap, :, :], second_patch[:overlap, :, :])

                    square_overlap = v_res2[:overlap, :, :] + h_res2[:, :overlap]
                    square_overlap = cv.boxFilter(square_overlap, -1, (2, 2))
                    x, y, z = np.where(square_overlap == square_overlap.max())

                    target_image[x_1 + x[0]:x_1 + height_of_block, y_1:y_1 + overlap] = v_res[x[0]:, :]
                    target_image[x_1:x_1 + overlap, y_1 + y[0]: y_1 + width_of_block] = h_res[:, y[0]:]

                    x, y, z = np.where(target_image[x_1: x_1 + height_of_block, y_1: y_1 + width_of_block] == 0)

                    target_image[x + x_1, y_1 + y] = second_patch[x, y]

                    target_image[x_1 + x[0]:x_1 + height_of_block, y_1:y_1 + overlap] = cv.boxFilter(
                        target_image[x_1 + x[0]:x_1 + height_of_block, y_1:y_1 + overlap], -1,
                        (3, 3))
                    target_image[x_1:x_1 + overlap, y_1 + y[0]: y_1 + width_of_block] = cv.boxFilter(
                        target_image[x_1:x_1 + overlap, y_1 + y[0]: y_1 + width_of_block], -1,
                        (3, 3))
