from utilities import *

src_image = cv.imread(base_images_address + "greek_ship.jpg")
template = cv.imread(base_images_address + "patch.png")

src_image = src_image.astype(np.float_)
template = template.astype(np.float_)

gray_template = cv.imread(base_images_address + "patch.png", 0)

Ib, Ig, Ir = cv.split(src_image)
Tb, Tg, Tr = cv.split(template)

new_template = cut_object_of_patch(gray_template, Tb)
result = normalized_cross_correlation(Ib, new_template)
result = apply_gamma_transformation(result, 5)

w, h = new_template.shape[0], new_template.shape[1]

draw_rectangle_around_detected_template(src_image, result, w, h, number_of_object)

cv.imwrite(base_results_address + "res03.jpg", src_image)
