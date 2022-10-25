from utilities import *

src_image = cv.imread(base_images_address + "flowers_blur.png")

unsharp = unsharp(src_image)
sharp = sharpe(src_image, .625)

cv.imwrite(base_results_address + "res02.jpg", sharp)
cv.imwrite(base_results_address + "res01.jpg", unsharp)
