r"""
file description:
This startup file tests the kmeans algorithm by applying it 
to random set of points. 
The user has to provide a set of initial center points!
"""


import clustering.kmeans_detailed as clust
import clustering.distance as distance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import clustering.testing as testing

#Definitions__________________________
# create 3 random sets for clustering
randomset1 = testing.randomset(5,10,-10)
randomset2 = testing.randomset(4,8,-10)
randomset3 = testing.randomset(3,5,-9)

""" combine to one set of points. 
This is necessary because kmeans takes as argument pos, 
a single np.array containing all points to be clustered.
Concatenation is done along axis 0."""
randomset_all = np.concatenate((randomset1, randomset2, randomset3), axis = 0)



# Here the clustering__________________________________________________________________________________
# specify accuracy of clustering
eps = 0.01
# specify maximum number of iterations
max_iter = 2000
# define array of centers to start with
centr = np.array([[2,-10], [5,-11], [8,-9]])

# kmeans clustering, return the final center points and the closest cluster for each point
centr, closest_cluster = clust.kmeans(randomset_all, centr, eps, max_iter, distance.dist_euclidean)

# Here the plots ___________________________________________________________________________________
# points before clustering. Each set is arranged in a square.
plt.subplot(2,2,2)
# scatterplot of all axis0, axis1-index 0 components against axis0, axis1-index1 components
plt.scatter(randomset_all[:,0], randomset_all[:,1])
plt.title('Points to be clustered')
plt.xlabel('x')
plt.ylabel('y')

# Set of points as predefined clusters. Each color corresponds to one set. 
plt.subplot(2,2,3)
# scatterplot of each set of points, 
# axis0, axis1-index 0 components against axis0, axis1-index1 components
plt.scatter(randomset1[:,0], randomset1[:,1])
plt.scatter(randomset2[:,0], randomset2[:,1])
plt.scatter(randomset3[:,0], randomset3[:,1])
# scatterplot of the center points, 
# axis0, axis1-index 0 components against axis0, axis1-index1 components
plt.scatter(centr[:,0], centr[:,1])
plt.title('Predefined clusters')
plt.xlabel('x')
plt.ylabel('y')

# clustered points. Each color corresponds to one cluster. 
plt.subplot(2,2,4)
r""" This for loop iterates over the center points. 
For each center point i, it fills the vector cluster_0 
with the points of randomset_all, for which the
closest center point is i. 
These points are then plotted as a scatter plot, with 
axis0, axis1-index 0 components against axis0, axis1-index1 components. 
The process is repeated for all center points, so that 
there are plotted as many clusters as center points assigned. 
"""
for i in range(len(centr)):
  cluster_0 = randomset_all[closest_cluster == i]
  plt.scatter(cluster_0[:,0], cluster_0[:,1])
plt.title('Points after applied clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



