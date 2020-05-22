
import numpy as np
import matplotlib.pyplot as plt
import math


#HERE Functions ________________________

#NAMING CONVENTION
# center or center point instead of centroid, 
# point, array
# amount of points instead of number of points
# components of a point instead of elements of a point
# elements of a vector
# computed instead of calculated

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
    # distance can be computed of this point to ALL the centers 
    # (and not only the centr.shape[0]th center).  
    i_c = 0
    # ele1 will be the input for the distance function. 
    # the distance function takes (ele1, ele2) as argument,
    # where ele1 is the point of the set pos at index i_pos, 
    # with all components.
    ele1 = pos[i_pos,:]
    # this steps computes the distance between the selected point
    # of the set and the center point of centr, at index i_c,
    # with all components. The output is distance_start: the distance of 
    # point of the set pos at index i_pos to the center point
    # of centr at index i_c.
    distance_start = dist(ele1, centr[i_c,:])
    # Here the first cluster gets assigned to the point of the set. 
    # The value of closest_cluster at the index i_pos should be 
    # the index of the corresponding center point in centr. 
    # In this way, the index i_pos of pos gets assigned to the 
    # index i_c of the center. printing closest_cluster at position
    # i_pos will output the index i_c of the center point in centr 
    # that is closest to the point of the set at i_pos. 
    closest_cluster[i_pos] = i_c
    # dist_point_cluster contains at index i_pos: distance_start, 
    # the distance of the point of a set to the center point. 
    # This array will be updated in the following while. 
    dist_point_cluster[i_pos] = distance_start
    # the counter for center is set to the next integer number. 
    # Now the above steps are repeated inside a new while.
    # This enables to compare the already computed distance 
    # between point of the set and center point at index i_c, 
    # with the other distances between 
    # point of the set and center point at index (i_c+1). 
    i_c=i_c+1
    # the iteration over the center points starts. 
    # It lasts until the index i_c is smaller than the amount 
    # of center points in axis0 direction (until 
    # all centers have been iterated over). 
    while i_c < centr.shape[0]:
      # ele1 is the input for the distance function. 
      # The steps above repeat. 
      ele1 = pos[i_pos,:]
      # the new distance gets stored in distance_compare, 
      # not in distance_start. This is for comparing the value 
      # in distance_compare with the value in distance_start. 
      distance_compare = dist(ele1, centr[i_c,:])
      # here is the comparison, the smallest distance is saved. 
      # if the new distance (distance between the point of the 
      # set and the next center in centr) is smaller than the 
      # start distance (distance between the point of the set 
      # and the previous center in centr), the value in distance_start 
      # gets overwritten by the value in distance_compare. 
      # Distance_start is the variable used by the next assignment. 
      if (distance_compare<distance_start) == True:
        distance_start = distance_compare
        # here the content of closest_cluster is updated. 
        # If the currently selected center is closer to the 
        # point than the previous center, the value at index
        # i_c gets overwritten by the value at the next index 
        # (remember: before starting this while loop, 
        # the counter i_c has been increased by 1). Axis1 is 
        # irrelevant, only axis0 is needed. 
        closest_cluster[i_pos] = i_c
        # here the content of dist_point_cluster is updated. 
        # If the assignment of the closest center changes, 
        # also the corresponding distance has to be updated. 
        # Remember: the content of distance_start has been 
        # overwritten with the content of distance_compare. 
        # Thus, here the new smallest distance gets assigned. 
        dist_point_cluster[i_pos] = distance_start
      
      # this is the counter for the second while, 
      # the iteration over the centers. The same procedure 
      # will be done for the next center point in centr, 
      # always taking distance_start as the reference distance 
      # for comparison. When the beginning condition is not met 
      # anymore, that i_c < amount of center points in centr, 
      # the while is not entered anymore. The smallest 
      # distance has been found and stored in dist_point_cluster
      # at the index of the point in pos. 
      # The corresponding center to the point of the set
      # has been found and stored in closest_cluster at
      # the index of the point in pos. 
      i_c = i_c+1
    # The while has iterated over all centers for the previous 
    # point. This counter increases the index i_pos by 1. 
    # In this way, the next point of the set is selected 
    # for computing its distance to all center points of centr.
    i_pos = i_pos+1
  # The algorithm determinates when i_pos is not < the 
  # amount of points of the set anymore. 
  # It means that every point has been already selected, 
  # and the closest center for that point determined. 
  # The function returns closest_cluster: the amount 
  # of elements inside this array is equal to the 
  # amount of points of the set. At closest_cluster[i_pos]
  # is the index of the center point in centr, 
  # that has been assigned to the point of the 
  # set at pos[i_pos,:]. 
  # Thus, centr[closest_cluster[i_pos],:] gives 
  # the centr point with ALL components that got assigned 
  # to point of the set at index pos[i_pos,:].
  # The function returns dist_point_cluster: 
  # the (smallest) distance between point of the set
  # at pos[i_pos,:] and the closest center point at 
  # centr[closest_cluster[i_pos],:]. 
  return closest_cluster, dist_point_cluster








# This function updates the center list centr
# input:
#   - pos: numpy.array as matrix containing the 
#   set of points. 
#       pos.shape[0] must be the amount of points. 
#       pos.shape[1] is the amount of components 
#       of each point.
#   - centr: numpy.array as matrix containing the 
#   set of centers of the clusters. 
#       centr.shape[0] must be the amount of 
#       centers. 
#       centr.shape[1] is the amount of components 
#       defining each center.
#   - closest_cluster: a numpy.array of length 
#   pos.shape[0] that contains the cluster number 
#   corresponding to each point in pos.

# output:
#   - centr: numpy.array as matrix containing the 
#   updated set of centers of the clusters.
#              Centr is now updated and ready 
#              for the next round of clustering. 
 
# In a clustering, centers are assigned to points of a set not
# only once, but multiple times. After each iteration 
# of the clustering, the center points are computed newly
#  as the arithmetic mean of all the points of the set that 
#  have been assigned to that center in the previous round 
#  of clustering. update_centr computes the new center 
# points of the centr array, for the next round of clustering. 
# It takes closest_cluster to find out which points of 
# the set pos should be taken for each arithmetic mean, 
# to compute the new center point as the arithmetic mean of 
# these points of the set for which pos[closest_cluster == i]. 
def update_centr(pos, centr, closest_cluster):
  # this iterator is a counter. iterator is an array
  # containing all integer numbers from 0 to the amount 
  # of points of the set pos.
  iterator = range(centr.shape[0])
  # iteration over all elements in iterator. This serves 
  # as a counter, to select all the indices at which 
  # closest_cluster contains the value i.  
  for i in iterator:
    # selector is an array containing all the indices at 
    # which closest_cluster contains the value i 
    # (the cluster number i). This can be used to select
    # all points in pos that have been assigned to this cluster. 
    selector = (closest_cluster == i)
    # the arithmetic mean can only be calculated if the 
    # center point has actually been assigned. 
    # It can happen that a center is not assigned.
    # This if condition checks if selector contains any values 
    # larger than 0 (if any center indices are inside). 
    # If the sum of all elements in selector
    # is > 0, then the center at index i has been assigned 
    # at least once.  
    if np.sum(selector) > 0:
      # cluster points contains all points that have 
      # been assigned to the center point at index i in centr 
      # (all the points that have been assigned
      # to that cluster i). The iteration is done along axis0, 
      # the components of each point of the set are selected with :. 
      cluster_points = pos[selector,:]
      # the arithmetic mean of the points of the set assigned 
      # to center i is 
      # computed with np.mean. Axis = 0 indicates the mean 
      # should be computed for each component separately. 
      # The array arithm_mean contains pos.shape[1]
      # amount of elements. 
      arithm_mean = np.mean(cluster_points, axis = 0)
    # if selector does not contain any value larger than 0 
    # (no point of the set has been assigned to center i), 
    # the else is entered. 
    else:
      # the arithmetic mean is faked as an empty array of 
      # the same amount of components as the normal 
      # arithmetic mean of pos.
      arithm_mean = np.zeros(centr.shape[1])
      # The array is filled with 'nan' to indicate no 
      # assignment has happened, and to prevent the program 
      # from crashing. Type float is chosen to not hinder the 
      # further algorithm which works with float. 
      arithm_mean = arithm_mean*float('nan')
    # The new center point at index i,: of centr is the 
    # computed arithmetic mean.
    centr[i,:] = arithm_mean
  # This algorithm proceeds for all elements in iterator, 
  # i.e. until all center points have been updated. 
  # The output is centr: filled with new center points 
  # (that can also be nan). 
  return centr



# This function calculates the sum of (distances 
# between each old center point and the new center point, 
# respectively).
# input: 
#   - centr: numpy.array as matrix containing the set 
#   of center points of the clusters. 
#       centr.shape[0] must be the number of centers. 
#       centr.shape[1] is the number of components
#        defining each center.
#   - centr_old: same specification of centr, 
#   contains the centr points of the previous clustering round.
#   - dist: a function that returns a positive 
#   real number given two arrays. 
#           It will be used for calculating the 
#           distance between two points. 
#           In this function, the points are 
#           numpy arrays of length pos.shape[1].
# output:
#   - new_eps: A float representing the sum of 
#   (distances between each old center point and the 
#   new center point, respectively).

# The accuracy function sets a limit when to stop the 
# clustering. It is defined as the amount that all 
# center points change from one clustering round to 
# another (as measured by a distance function). It 
# computes the sum of individual changes from each 
# center point to its old center point. 
# Once the change is lower than a specified accuracy 
# eps, the algorithm should terminate. 
# centr_old is specified in the function julia_kmeans(), 
# it is the array of center points of the previous round 
# of clustering. 

def accuracy(centr, centr_old, dist):
  # new_eps is instantiated here. The value should 
  # be set to 0 to avoid interference with the specified 
  # eps. 
  new_eps = 0
  # this iterator iterates over all center points in center. 
  iterator = range(centr.shape[0])
  for i in iterator:
    # The dist_centr_old contains 1 value, the distance between 
    # the old center point at index i in centr_old 
    # and the updated center point at index i in centr.
    dist_centr_old = dist(centr_old[i,:], centr[i,:])
    # This if checks if dist_centr_old contains a distance. 
    # If a center point has not been assigned, dist_centr_old
    # will not contain a distance. To prevent crashing of the 
    # algorithm, this dist_centr_old will be ignored for computing
    # the new accuracy new_eps. If dist_centr_old does NOT contain
    # nan, the if is entered. 
    if math.isnan(dist_centr_old) == False: 
      # The new accuracy is the sum of all (distances of each 
      # center point to its old center point). In each round of 
      # iteration inside the for, the accuracy value for the 
      # next center point is added. 
      new_eps = new_eps + dist_centr_old
  # the new_eps accuracy after the center changed is output.  
  return new_eps






# This functions makes a kmeans clustering. 
# input: 
# 
#   - pos: numpy.array as matrix containing the set of points. 
#       pos.shape[0] must be the amount of points. 
#       pos.shape[1] is the amount of components defining 
#       each point.
#   - centr: numpy.array as matrix containing the set 
#   of centers of the clusters. 
#       centr.shape[0] must be the amount of centers. 
#       centr.shape[1] is the amount of components defining 
#       each center.
#   - eps: A float representing the required sum of 
#   (distances between each old center and the new 
#   center, respectively).
#         when this accuracy is reached, clustering stops. 
#   - max_iter: an integer number specifying the 
#   maximum number of clustering iterations.
#   - dist: a function that returns a positive 
#   real number given two arrays. 
#           It will be used for calculating the 
#           distance between two points. 
#           In this function, the points are numpy 
#           arrays of length pos.shape[1].

# output:
#   - centr: numpy.array as matrix containing 
#   the set of centers of the clusters. 
#       centr.shape[0] must be the amount of centers. 
#       centr.shape[1] is the amount of components defining 
#       each center. 
#   - closest_cluster: a numpy.array of length pos.shape[0] 
#   that contains the cluster number corresponding to 
#   each point in pos.

# this functions makes a kmeans clustering, with k centers, 
# and the mean as the criterium for finding new centers. 
# the arguments have been specified above. New is max_iter: 
# an integer number to specify the maximum number the while 
# loop should execute.
def julia_kmeans(pos, centr, eps, max_iter, dist):
  # m is the counter for max_iter, it counts the round 
  # of clustering.
  m = 0
  # eps is the required accuracy to stop the algorithm. 
  # In the first round, the variable new_eps must contain a 
  # larger value than eps, otherwise the while is not entered
  # and the clustering is not done. Inside the while, new_eps
  # is updated. For simplification, the value of eps + 1 
  # has been chosen as the starting value for new_eps. 
  new_eps = eps + 1
  # this list contains all values of new_eps after each cluster round. 
  eps_track = []
  # here is the copy of centr, filled with zeros. It will 
  # contain a copy of the center points of centr after each 
  # clustering round. 
  centr_old = np.zeros(centr.shape)
  # the operations done on centr necessitate it to be float. 
  centr = centr.astype(float)
  # here the kmeans iteration starts. the continuation criterium 
  # is that the new accuracy is still larger than the specified 
  # accuracy eps, and the round of clustering m is still smaller 
  # than the specified number of rounds max_iter. If new_eps 
  # becomes smaller or equal than the specified accuracy, 
  # or the number of rounds specified has been reached, 
  # the while is not entered anymore. 
  while new_eps > eps and m < max_iter:
    # Here is the function called that assigns to each point of 
    # the set pos the center point in centr which is closest to 
    # that point (which has smallest distance to that point). 
    closest_cluster, dist_point_cluster = assign_closest_cluster(pos, centr, dist)
    # here the values of centr get copied into centr_old.
    centr_old[:,:] = centr[:,:]

    # Here, is the function called that updates the centr values. 
    centr = update_centr(pos, centr, closest_cluster)
    # Here is the function called that computes the accuracy 
    # the change in center points between centr and centr_old)
    new_eps = accuracy(centr, centr_old, dist)
    # Here the list eps_track gets filled with new_eps.
    eps_track.append(new_eps)
    # The counter is increased by one. The while proceeds from the 
    # beginning, checking first if the conditions are still met, and 
    # then entering the while or exiting the function.
    m = m + 1
    #print(new_eps)

  # this kmeans function contains a visual output to check how the 
  # accuracy changes. It plots eps_track. 
  plt.figure(figsize = (35, 35))
  plt.subplot(2,2,1)
  # marker s means square, ls = '' is no line. 
  plt.plot(eps_track, marker="s", ls = '')
  plt.title('Change of accuracy eps')
  plt.yscale('log')
  plt.xlabel('Number of clustering round')
  plt.ylabel('eps')

  # finally, the function kmeans returns the final array of center points 
  # after clustering, and the array containing the number (as index) of 
  # the center point that got assigned to each point of the set in pos
  # (at the same index as in closest_cluster). 
  # Happy clustering! 
  return centr, closest_cluster
