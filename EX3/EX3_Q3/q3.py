from utilities import *

num_of_clusters, scale = get_inputs()

image = cv.imread(base_images_address + 'slic.jpg')

rgb_image = image
lab_image = cv.bilateralFilter(image, -1, 150, 10)
lab_image = cv.cvtColor(lab_image, cv.COLOR_BGR2LAB)

edges = get_edges(lab_image)

lab_scaled_image = resize_image(lab_image, scale)
rgb_scaled_image = resize_image(rgb_image, scale)
scaled_edges = resize_image(edges, scale)

data = np.full((lab_scaled_image.shape[0], lab_scaled_image.shape[1], 2), np.inf)

cluster_centers, S = initialize_cluster_centers(lab_scaled_image, num_of_clusters, scaled_edges, indexes)

f = np.ndarray((2 * S, 2 * S, 2))
for i in range(2 * S):
    for j in range(2 * S):
        f[i, j] = [int(i - S), int(j - S)]

segments = slic(cluster_centers, lab_scaled_image, S, f, data)

draw_boundaries(segments, rgb_scaled_image)

cv.imwrite(base_results_address + 'test.jpg', rgb_scaled_image)
