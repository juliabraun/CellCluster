
import numpy as np
import matplotlib.pyplot as plt
import math


#HERE Functions ________________________

#NAMING CONVENTION
# center, point, array


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
# distance functions distance.dist_manhattan() and distance.dist_euclidean().
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
  # and b is the amount of array elements in axis1 direction
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
  # The new array closest_cluster should contain at each index the 
  # integer number of the centr that corresponds to the point of 
  # the image array at the same index 
  # (here: position x_p,y_p,...n_p of pixel). Then, writing 
  # "print(closest_cluster[index])" with index replaced by the 
  # index number i, will output the center number that the data point
  # in pos with index i got assigned to.
  # Index must be integer, therefore the object type is fixed 
  # as int with np.int32. 
  # closest_cluster is an array filled with zeros, containing 
  # the same amount of array elements in axis0 direction as the 
  # image array pos (the array of data points, here: the np.array of 
  # image points). Thus, the amount of indices of closest_cluster is 
  # equal to the amount of indices of pos.
  closest_cluster = np.int32(np.zeros([pos.shape[0]]))
  # The new array dist_point_cluster should contain at each index 
  # the computed distance of the point to its CLOSEST center. 
  # The following algorithm will find out, which center is the 
  # closest for each data point (here: the distance of each pixel 
  # to its closest center). Then, writing 
  # "print(dist_point_cluster[index])" with index replaced by the 
  # index number i, will output the distance between the data point
  # in pos with index i for axis0 (pos[i,:], does not correspond to 
  # the pixel position in the original image, but to the pixel 
  # position in the tresholded point array pos that contains the 
  # positions of highest intensity) and the closest center 
  # (stored in centr at index i).
  # Distance is a float, therefore the object type is not fixed since 
  # np.zeros() produces arrays filled with float per default. 
  dist_point_cluster = np.zeros([pos.shape[0]])
  # Two counters are needed to calculate the distance of each
  # point of the set (here: each pixel of highest value, 
  # indicating one cell per point) to each center. 
  # One counter iterates over the indices of pos 
  # (one of the selected image point after the other).
  # For each of these points, one counter is needed to iterate 
  # over all centers (one center after the other). Hence, 
  # there is an iteration inside an iteration. 
  # The counter for iteration over the points of the set pos 
  # is called i_pos. The counter for the iteration over the
  # centers is called i_c. Any variable has to be instanciated 
  # first before it can be used. Instanciation means defining 
  # variable by writing it down and assigning a value to it. 
  # The counters get instanciated with the value 0 because the 
  # counting should start at index 0.
  i_pos = 0
  # The iteration goes from index i_pos, 0 in the first round, 
  # to the index pos.shape[0] (the amount of points of the set 
  # pos minus 1). The minus 1 is due to the ``zero indexing'' 
  # python: indexing does not go from 1:end, but from 0:(end-1).
  # the ``:'' indicate a distance to be gapped. 
  while i_pos < pos.shape[0]:
    # the counter for iteration over the centers, i_c, gets 
    # instantiated inside the first while loop. In this way, 
    # every time, the next point of the set is reached, 
    # the counter over the centers is reset to 0. Then, the
    # distance can be computed of this point to all the centers 
    # (and not only the centr.shape[0]th center).  
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

