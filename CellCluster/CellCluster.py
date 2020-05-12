import clustering.kmeans_detailed as clust
import numpy as np

#HERE the code__________________________
pos = np.array([[10, 32], [12, 31], [15, 30], [20, 0], [21, -2]])
eps = 1.0

max_iter = 10

centr = np.array([[20,3], [5,3], [0, 10]])

clust.julia_kmeans(pos, centr, eps, max_iter)
