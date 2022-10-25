import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth

base_images_address = 'images/'
base_results_address = 'results/'

src_image = cv2.imread(base_images_address + 'park.jpg')
src_image_shape = src_image.shape

# Resize image
scale = 20

width = int(src_image_shape[1] * scale / 100)
height = int(src_image_shape[0] * scale / 100)
dim = (width, height)

image = cv2.resize(src_image, dim, interpolation=cv2.INTER_AREA)

# Converting image into array of dimension [nb of pixels in src_image, 3]
# based on r g b intensities
flatImg = np.reshape(image, [-1, 3])

# Estimate bandwidth for mean shift algorithm
bandwidth = estimate_bandwidth(flatImg, quantile=0.05, n_samples=100)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

# Performing mean shift on flatImg
ms.fit(flatImg)

# (r,g,b) vectors corresponding to the different clusters after mean shift
labels = ms.labels_

cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print('number of labels: ' + str(n_clusters_))

segmentedImg = cluster_centers[np.reshape(labels, image.shape[:2])]
labels = labels.reshape(image.shape[:2])

cv2.imwrite(base_results_address + 'res04.jpg', segmentedImg)
