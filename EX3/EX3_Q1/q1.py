from utilities import *
from pandas import DataFrame
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

points_file_data = open(points_file_name, 'r')

num_of_points = int(points_file_data.readline())
data = {'x': [], 'y': []}

ma = load_data(points_file_data, num_of_points, data)
data_frame = DataFrame(data, columns=['x', 'y'])

k_means = KMeans(n_clusters=2).fit(data_frame)
centroids = k_means.cluster_centers_

plt.scatter(data_frame['x'], data_frame['y'], c=k_means.labels_.astype(float), s=50, alpha=0.5)
plt.show()
