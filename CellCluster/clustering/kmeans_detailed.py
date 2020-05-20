
import numpy as np
import matplotlib.pyplot as plt
import math


#HERE Functions ________________________

#NAMING CONVENTION
# center, point, array



# - this function calculates the manhattan distance between the 
#   positions of two points 
#   (e.g. position x_p,y_p,...n_p of pixel and 
#   position x_p,y_p,...n_p of center). 
#           It returns a positive real number given two arrays. 
# - input: 
#     - ele1, ele2: numpy arrays of length pos.shape[1].
# - output:
#     - sum: the manhattan distance as sum of the distances of 
#     the point components: |x1 - x2| + |y1 - y2|+ ... + |n1-n2|.

def dist_manhattan(ele1, ele2):
  # ele1 is a point (here: pixel), ele2 is a second point 
  #   (here: the center).
  # np.abs returns the absolute value (negative values become 
  #   positive). 
  # dist1 therefore is a array with index-wise differences, 
  #   it contains: 
  #   x1-x2 at index 0,  y1-y2 at index 1, ..., n1-n2 at 
  #     index n-1. 
  # The letter indicates the component (x,y,...n), 
  # the number indicates the point (1: pixel, 2: center).
  dist1 = np.abs(ele1 - ele2)
  # the manhattan distance is defined as the sum of the 
  # components of dist1: |x1 - x2| + |y1 - y2|+ ... + |n1-n2|. 
  # np.sum makes index-wise sum over the values of all indices. 
  sum = np.sum(dist1)
  # the output is sum: the manhattan distance between one point 
  #   of the dataset (here: position x,y,...n of pixel) 
  #   and the second point (here: position x,y,...n of center). 
  return sum



# - this function calculates the euclidean distance of the 
#     positions of two points (e.g. position x_p,y_p,...n_p of pixel 
#     and position x_c,y_c,...n_c of center). 
#     It returns a positive real number given two arrays. 
# - input: 
#     - ele1, ele2: numpy arrays of length pos.shape[1].
# - output:
#     - sum: the euclidean distance as squareroot of the 
#     sum of the squared differences between the components of 
#     two points: sqrt((|x1 - x2|)^2 + (|y1 - y2|)^2 + ... + (|n1 - n2|)^2).

def dist_euclidean(ele1, ele2):
  # ele1 is a point, ele2 is a second point 
  #   (here: the center).
  # np.abs returns the absolute value 
  #   (negative values become positive). 
  # dist2 therefore is a array with index-wise differences, 
  #   it contains: x1-x2 at index 0, y1-y2 at index 1, ..., 
  #   n1-n2 at index n-1. 
  #   The letter indicates the component (x,y,...n), 
  #   the number indicates the point (1: pixel, 2: center)
  # The name dist2 has been chosen to avoid confusion with 
  #   dist1 from dist_manhattan().
  dist2 = np.abs(ele1 - ele2)
  # squared is dist2 to the power of 2, thus an array with 
  # the index-wise differences squared: (x1-x2)^2 and (y1-y2)^2
  squared = np.power(dist2, 2)
  # The Euclidean distance is defined as the squareroot of the
  # sum of the squared differences between the components of 
  # two points = the hypothenusis in pythagorean theorem. 
  # np.sum makes index-wise sum over the values of all indices. 
  #   It gives a number representing the sum of the squared 
  #   differences between the same component of two different 
  #   points. 
  sum = np.sum(squared)
  # np.sqrt returns the squareroot of the input value, 
  # which is the euclidean distance between one point 
  # (position x,y,...n of pixel)and the second point
  # (position x,y,...n of center).
  root = np.sqrt(sum)
  return root



# This function calculates the closest cluster for each 
# point inside a set of points.
# centers centr of the clusters must be provided.
# input: 
#   - pos: numpy.array as matrix containing the set of points. 
#       pos.shape[0] must be the number of points. 
#       pos.shape[1] is the number of elements defining each point. 
#   - centr: numpy.array as matrix containing the set of centroids 
#     of the clusters. 
#       centr.shape[0] must be the number of centroids. 
#       centr.shape[1] is the number of elements defining each 
#       centroid.
#   - dist: a function that returns a positive real number given 
#     two arrays. It will be used for calculating the distance 
#     between two points. In this function, the points are numpy 
#     arrays of length pos.shape[1].
# output: 
#   - closest_cluster: a numpy.array of length pos.shape[0] 
#   that contains the cluster number corresponding to each point 
#   in pos.
#   - dist_point_cluster: a numpy.array of length pos.shape[0] 
#   that contains the distance dist of each point in pos to 
#   its closest center.
# WARNING: 
#   - centr.shape[1] must be equal to pos.shape[1]. 
#    

# This functions finds for each point in pos the corresponding 
# center in centr. Corresponding means, this centr is the 
# closest to the point. The smaller the value of the computed
# distance, the closer are two points. The algorithm calculates 
# the distance of each position of one point 
# (here: position x_p,y_p,...n_p of pixel) to the position of the 
# second point (here: position x_c,y_c,...n_c of center). 
# The calculation is done component-wise, as specified in the
# distance functions dist_manhattan() and dist_euclidean().
# Thus, every data point is compared with every cluster. 
# Pos is the array of data points, here: the np.array of 
# image points. Any dimensional array can be used, but the 
# number of components of one point 
#  (here: position x_p,y_p,...n_p of pixel)
# must be equal to the number of components of the second point 
# (here: position x_c,y_c,...n_c of center). 
# This is checked in the first line of the function.
def assign_closest_cluster(pos, centr, dist):
  # here is a check if the amount of components of one point 
  # (here: position x_p,y_p,...n_p of pixel)
  # are equal to the amount of components of the second point 
  # (here: position x_c,y_c,...n_c of center). 
  # The method .shape return the dimension of an object, 
  # as (a,b), where a is number of array elements in axis0 direction 
  #   (here: the amount of pixels of the image, 
  #   the amount of points of the image array, 
  #   the amount of centers specified),
  # and b is the number of array elements in axis1 direction
  #   (here: the amount of components of each pixel of the image, 
  #   the amount of components of each point of the image array
  #   all points have to have the same amount of components, 
  #   the amount of components of each center).
  # The values at index a,b can be accessed with 
  #   a = centr[0] and b = centr[1]. 
  # If the amount of components of one point (the pixel) is not 
  # equal to the amount of components of the second point 
  # (the center), the closest cluster cannot be assigned. 
  # In this case, the algorithm warns the user that shape[1] of
  # pos and centr are not equal, i.e. the amount of array elements
  # in axis1 direction are not equal. Since the program would crash
  # without equal shape[1] of one point (pixel) and the second point
  # (center), the algorithm should not execute. 
  if(centr.shape[1] != pos.shape[1]):
    print("error in \"def closest_cluster(pos, centr)\" shape[1] \
    of pos and centr incompatible")
    # 1. Return stops the execution. The function is not executed 
    # further. Without return, the function would continue to 
    # execute, albeit the shape[1] of pos and centr are not equal. 
    # 2. The structure of this "error return" has to match the 
    # structure of the "normal" function return. Here, 
    # assign_closest_cluster() returns TWO objects: 
    # closest_cluster and dist_point_cluster. Hence, also 
    # "error return" has to return TWO objects. 
    # These can be chosen arbitrary, but should indicate that 
    # an error has happened. In case of error, executing 
    # "print(closest_cluster)" will here output: "-1". 
    # An intact closest_cluster should contain the index of 
    # centr which corresponds to each point 
    # (here: the center that is closest to each pixel). 
    # "-1" is not an index of the centr array. Hence, something is 
    # wrong with this function. 
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
#   - dist: a function that returns a positive real number given two arrays. 
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
#   - dist: a function that returns a positive real number given two arrays. 
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
    print(new_eps)


#__________________________________________________________________________________
#Here the code
  #print(eps_track)

  plt.figure(figsize = (35, 35))
  plt.subplot(2,2,1)
  plt.plot(eps_track, marker="s", ls = '')
  plt.title('Change of accuracy eps')
  plt.yscale('log')
  plt.xlabel('Number of clustering round')
  plt.ylabel('eps')
  return centr, closest_cluster


#d changes the scaling from [0,1) to [0,d). b translates the set of points by px + bx, py + by. 

