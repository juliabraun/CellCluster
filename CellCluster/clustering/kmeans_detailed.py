
import numpy as np
import matplotlib.pyplot as plt
import math


#HERE Functions ________________________


# - this function calculates the manhattan distance between two points (e.g. point and cluster center). 
#           It returns a positive real number given two vectors. 
# - input: 
#     - ele1, ele2: numpy arrays of length pos.shape[1].
# - output:
#     - sum: the manhattan distance as sum of the distances of the point components: |x1 - x2| + |y1 - y2|.

def dist_manhattan(ele1, ele2):
  dist1 = np.abs(ele1 - ele2)
  sum = np.sum(dist1)
  return sum



def dist_euclidean(ele1, ele2):
  dist2 = np.abs(ele1 - ele2)
  squared = np.power(dist2, 2)
  sum = np.sum(dist2)
  root = np.sqrt(sum)
  return root




# This function calculates the closest cluster for each point inside a set of points.
# Centroids centr of the clusters must be provided.
# input: 
#   - pos: numpy.array as matrix containing the set of points. 
#       pos.shape[0] must be the number of points. 
#       pos.shape[1] is the number of elements defining each point. 
#   - centr: numpy.array as matrix containing the set of centroids of the clusters. 
#       centr.shape[0] must be the number of centroids. 
#       centr.shape[1] is the number of elements defining each centroid.
#   - dist: a function that returns a positive real number given two vectors. 
#           It will be used for calculating the distance between two points. 
#           In this function, the points are numpy arrays of length pos.shape[1].
# output: 
#   - closest_cluster: a numpy.array of length pos.shape[0] that contains the cluster 
#                   number corresponding to each point in pos.
#   - dist_point_cluster: a numpy.array of length pos.shape[0] that contains the distance dist 
#                   of each point in pos to its closest centroid
# WARNING: 
#   - centr.shape[1] must be equal to pos.shape[1]. 
#    

def assign_closest_cluster(pos, centr, dist):
  if(centr.shape[1] != pos.shape[1]):
    print("error in \"def closest_cluster(pos, centr)\" shape of pos and centr incompatible")
    return -1, -1
  closest_cluster = np.int32(np.zeros([pos.shape[0]]))
  dist_point_cluster = np.zeros([pos.shape[0]])
  i_pos = 0
  while i_pos < pos.shape[0]:
    i_c = 0
    ele1 = pos[i_pos,:]
    sum = dist(ele1, centr[i_c,:])
    closest_cluster[i_pos] = i_c
    dist_point_cluster[i_pos] = sum
    i_c=i_c+1
    while i_c < centr.shape[0]:
      ele1 = pos[i_pos,:]
      sumnew = dist(ele1, centr[i_c,:])
  
      if (sumnew<sum) == True:
        sum = sumnew
        closest_cluster[i_pos] = i_c
        dist_point_cluster[i_pos] = sum
      
    
      i_c = i_c+1
    i_pos = i_pos+1
  return closest_cluster, dist_point_cluster


# This function updates the centroid list centr
# input:
#   - pos: numpy.array as matrix containing the set of points. 
#       pos.shape[0] must be the number of points. 
#       pos.shape[1] is the number of elements defining each point.
#   - centr: numpy.array as matrix containing the set of centroids of the clusters. 
#       centr.shape[0] must be the number of centroids. 
#       centr.shape[1] is the number of elements defining each centroid.
#   - closest_cluster: a numpy.array of length pos.shape[0] that contains the cluster 
#                   number corresponding to each point in pos.

# output:
#   - centr: numpy.array as matrix containing the updated set of centroids of the clusters.
#              Centr is now updated and ready for the next round of clustering. 
  
def update_centr(pos, centr, closest_cluster):
  iterator = range(centr.shape[0])
  for i in iterator:
    selector = (closest_cluster == i)
    if np.sum(selector) > 0:
      cluster_points = pos[selector,:]
      arithm_mean = np.mean(cluster_points, axis = 0)
    else:
      arithm_mean = np.zeros(centr.shape[1])
      arithm_mean = arithm_mean*float('nan')
    centr[i,:] = arithm_mean
  return centr



# This function calculates the sum of (distances between each old centroid and the new centroid, respectively).
# input: 
#   - centr: numpy.array as matrix containing the set of centroids of the clusters. 
#       centr.shape[0] must be the number of centroids. 
#       centr.shape[1] is the number of elements defining each centroid.
#   - centr_old: same specification of centr, contains the values of the previous clustering round.
#   - dist: a function that returns a positive real number given two vectors. 
#           It will be used for calculating the distance between two points. 
#           In this function, the points are numpy arrays of length pos.shape[1].
# output:
#   - new_eps: A float representing the sum of (distances between each old centroid and the new centroid, respectively).
def accuracy(centr, centr_old, dist):
  new_eps = 0
  iterator = range(centr.shape[0])
  for i in iterator:
    dist_centr_old = dist(centr_old[i,:], centr[i,:])
    if math.isnan(dist_centr_old) == False: 
      new_eps = new_eps + dist_centr_old
  return new_eps






# This functions makes a kmeans clustering. 
# input: 
# 
#   - pos: numpy.array as matrix containing the set of points. 
#       pos.shape[0] must be the number of points. 
#       pos.shape[1] is the number of elements defining each point.
#   - centr: numpy.array as matrix containing the set of centroids of the clusters. 
#       centr.shape[0] must be the number of centroids. 
#       centr.shape[1] is the number of elements defining each centroid.
#   - eps: A float representing the required sum of (distances between each old centroid and the new centroid, respectively).
#         when this accuracy is reached, clustering stops. 
#   - max_iter: an integer number specifying the maximum number of clustering iterations.
#   - dist: a function that returns a positive real number given two vectors. 
#           It will be used for calculating the distance between two points. 
#           In this function, the points are numpy arrays of length pos.shape[1].

# output:
#   - centr: numpy.array as matrix containing the set of centroids of the clusters. 
#       centr.shape[0] must be the number of centroids. 
#       centr.shape[1] is the number of elements defining each centroid. 
#   - closest_cluster: a numpy.array of length pos.shape[0] that contains the cluster 
#                   number corresponding to each point in pos.

def julia_kmeans(pos, centr, eps, max_iter, dist):
  m = 0
  new_eps = eps + 1
  eps_track = []

  centr_old = np.zeros(centr.shape)
  centr = centr.astype(float)
  while new_eps > eps and m < max_iter:
    # find cluster assignment
    closest_cluster, dist_point_cluster = assign_closest_cluster(pos, centr, dist)
    centr_old[:,:] = centr[:,:]
    #print(centr_old)

    # update the centroid values
    centr = update_centr(pos, centr, closest_cluster)
    # compute accuracy (change in cluster position)
    new_eps = accuracy(centr, centr_old, dist)
    eps_track.append(new_eps)
    m = m + 1

  #print(eps_track)

  plt.figure(figsize = (35, 35))
  plt.subplot(2,2,1)
  plt.plot(eps_track, marker="s", ls = '')
  plt.title('Change of accuracy eps')
  plt.xlabel('Number of clustering round')
  plt.ylabel('eps')
  return centr, closest_cluster


#d changes the scaling from [0,1) to [0,d). b translates the set of points by px + bx, py + by. 

