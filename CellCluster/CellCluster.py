import clustering.kmeans_detailed as clust
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#HERE the code__________________________



#d changes the scaling from [0,1) to [0,d). b translates the set of points by px + bx, py + by. 


def box(d,b,c):
  box1 = np.random.rand(2000,2) - 0.5
  box1 = d*box1
  box1[:,0] = box1[:,0] + b
  box1[:,1] = box1[:,1] + c
  return box1

box1 = box(5,10,-10)
box2 = box(4,8,-10)
box3 = box(3,5,-9)

box_all = np.concatenate((box1, box2, box3), axis = 0)



#Here the clustering__________________________________________________________________________________
eps = 0.01
max_iter = 2000

centr = np.array([[2,-10], [5,-11], [8,-9]])

centr, closest_cluster = clust.julia_kmeans(box_all, centr, eps, max_iter, clust.dist_euclidean)

plt.subplot(2,2,2)
plt.scatter(box_all[:,0], box_all[:,1])
plt.title('Points to be clustered')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(2,2,3)
plt.scatter(box1[:,0], box1[:,1])
plt.scatter(box2[:,0], box2[:,1])
plt.scatter(box3[:,0], box3[:,1])
plt.scatter(centr[:,0], centr[:,1])
plt.title('Predefined clusters')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(2,2,4)
for i in range(len(centr)):
  cluster_0 = box_all[closest_cluster == i]
  plt.scatter(cluster_0[:,0], cluster_0[:,1])

plt.title('Points after applied clustering')
plt.xlabel('x')
plt.ylabel('y')

plt.show()



