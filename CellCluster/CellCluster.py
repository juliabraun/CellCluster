import clustering.kmeans_detailed as clust
import clustering.distance as distance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import clustering.testing as testing

#HERE the code__________________________



#d changes the scaling from [0,1) to [0,d). b translates the set of points by px + bx, py + by. 
# create a test set of point clouds confined in a box
randomset1 = testing.randomset(5,10,-10)
randomset2 = testing.randomset(4,8,-10)
randomset3 = testing.randomset(3,5,-9)
randomset_all = np.concatenate((randomset1, randomset2, randomset3), axis = 0)



#Here the clustering__________________________________________________________________________________
eps = 0.01
max_iter = 2000

centr = np.array([[2,-10], [5,-11], [8,-9]])

centr, closest_cluster = clust.kmeans(randomset_all, centr, eps, max_iter, distance.dist_euclidean)

plt.subplot(2,2,2)
plt.scatter(randomset_all[:,0], randomset_all[:,1])
plt.title('Points to be clustered')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(2,2,3)
plt.scatter(randomset1[:,0], randomset1[:,1])
plt.scatter(randomset2[:,0], randomset2[:,1])
plt.scatter(randomset3[:,0], randomset3[:,1])
plt.scatter(centr[:,0], centr[:,1])
plt.title('Predefined clusters')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(2,2,4)
for i in range(len(centr)):
  cluster_0 = randomset_all[closest_cluster == i]
  plt.scatter(cluster_0[:,0], cluster_0[:,1])

plt.title('Points after applied clustering')
plt.xlabel('x')
plt.ylabel('y')

plt.show()



