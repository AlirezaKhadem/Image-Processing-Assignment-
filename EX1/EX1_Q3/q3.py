from utilities import *

image_name = get_image_name()
image = cv.imread(image_base_file_address + image_name, 0)

first_channel, second_channel, third_channel = split_image_horizontally(image, 3)

edf_channel = edge_detection_filter(first_channel)
eds_channel = edge_detection_filter(second_channel)
edt_channel = edge_detection_filter(third_channel)

fc_min_location = ssd(edf_channel, edf_channel, 2)
sc_min_location = ssd(edf_channel, eds_channel, 2)
tc_min_location = ssd(edf_channel, edt_channel, 2)

T2 = np.float32([[1, 0, sc_min_location[0] - fc_min_location[0]], [0, 1, sc_min_location[1] - fc_min_location[1]]])
T3 = np.float32([[1, 0, tc_min_location[0] - fc_min_location[0]], [0, 1, tc_min_location[1] - fc_min_location[1]]])

second_channel = cv.warpAffine(second_channel, T2, second_channel.shape[::-1])
third_channel = cv.warpAffine(third_channel, T3, third_channel.shape[::-1])

result = cv.merge((first_channel, second_channel, third_channel))

condition1 = (result[:, :, 0] > 253 )
condition2 = (result[:, :, 1] > 253)
condition3 = (result[:, :, 2] > 253)

condition11 = np.repeat(condition1, 3)
condition21 = np.repeat(condition2, 3)
condition31 = np.repeat(condition3, 3)

condition1 = condition11.reshape((condition1.shape[0], condition1.shape[1], 3))
condition2 = condition21.reshape((condition2.shape[0], condition2.shape[1], 3))
condition3 = condition31.reshape((condition3.shape[0], condition3.shape[1], 3))

result = np.where(condition1, 0, result)
result = np.where(condition2, 0, result)
result = np.where(condition3, 0, result)

result = result.astype(np.uint8)
cv.imwrite(result_base_file_address + "res04.jpg", result)
