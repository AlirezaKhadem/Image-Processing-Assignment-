from utilities import *

source_image = cv.imread(base_address_images + '1.source.jpg')
target_image = cv.imread(base_address_images + '1.target.jpg')
mask = cv.imread(base_address_images + 'mask.jpg', cv.IMREAD_GRAYSCALE) // 255  # binary mask

result = blend(warp_source_image(source_image, delta_x, delta_y, target_image.shape), target_image, mask)

cv.imwrite(base_address_results + 'res1.jpg', result)
