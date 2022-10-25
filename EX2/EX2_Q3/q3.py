from utilities import *

src_image = cv.imread(base_images_address + 'books.jpg')

# [[740, 358], [709, 154], [465, 410], [427, 205]] res04.jpg
# [[206, 664], [394, 600], [107, 381], [285, 318]] res05.jpg
# [[969, 813], [1099, 608], [668, 621], [796, 423]] res06.jpg

data_04 = np.load('04')
data_05 = np.load('05')
data_06 = np.load('06')

result_04 = warp_perspective(src_image, data_04)
result_05 = warp_perspective(src_image, data_05)
result_06 = warp_perspective(src_image, data_06)

cv.imwrite(base_result_address + 'res04.jpg', result_04)
cv.imwrite(base_result_address + 'res05.jpg', result_05)
cv.imwrite(base_result_address + 'res06.jpg', result_06)
