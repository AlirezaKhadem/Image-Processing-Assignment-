from copy import deepcopy
from skimage.segmentation import felzenszwalb
from utilities import *

image = cv.imread(base_images_address + 'birds.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
image = resize_image(image, 50)
result = deepcopy(image)

segments = felzenszwalb(image, scale=149, sigma=2.50, min_size=150)

print('number of clusters: ' + get_num_of_clusters(segments).__str__())

average_color(image, segments)

cv.namedWindow('window', cv.WINDOW_NORMAL)
cv.setMouseCallback('window', mouse_handler, {"image": image})
while True:
    cv.imshow('window', image)
    key = cv.waitKey(20)
    if key == 27:
        break

from utilities import sample

result = np.where(image == image[sample[0], sample[1], 0], 255, image)

cv.imwrite(base_results_address + 'res08.jpg', result)
