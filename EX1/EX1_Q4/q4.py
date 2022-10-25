from utilities import *

original_image = cv.imread(image_base_file_address + 'Dark.jpg')
target_image = cv.imread(image_base_file_address + 'pink.jpg')

result = histogram_specified(original_image, target_image)

cv.imwrite(result_base_file_address + 'res06.jpg', result)
