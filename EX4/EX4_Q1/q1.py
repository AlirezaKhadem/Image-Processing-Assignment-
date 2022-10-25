from utilities import *

src_image = cv.imread(images_address_base + 'texture1.jpg')
target_image = np.zeros((2700, 2700, 3))

texture_synthesis(src_image, target_image, 150, 150, 55)
cv.imwrite(results_address_base + 'res01.jpg', target_image[:2500, :2500])
