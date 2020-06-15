# ---- INFO -----
# This is the version with extended comments. For concise comments,
# see file kmeans_detailed.py.
# ---------------

# HERE imports
import numpy as np
import matplotlib.pyplot as plt
import math

# HERE File description ======================
# This file contains functions for performing a kmeans 
# clustering. 


# NAMING convention 
# a cluster center is called: CENTER or center point, 
# AMOUNT of points instead of number of points
# COMPONENTS of a point instead of elements of a point
# elements of a vector
# computed instead of calculated


# WARNINGS to the variables used:
# pos
#   - pos does not need to be a 2D np.array of the original image. 
#   - pos can also contain a subset of the original image, 
#     e.g. the position of the pixel with highest intensity values. 
# ==================================================

# HERE function description ------------------------
# Function START assign_closest_cluster() ***************************************

# This functions finds for each point in pos the corresponding 
# center in centr. Corresponding means, this centr is the 
# closest to the point. The smaller the value of the computed
# distance, the closer are two points. The algorithm calculates 
# the distance of each position of one point 
# (here: point of pos) to the position of the 
# second point (here: center point of center).
# The user provides his own distance function.  
# The calculation may be done component-wise, as specified in the
# distance functions distance.dist_manhattan() and distance.dist_euclidean(), 
# which are left as examples in file distance.py. 

# Any-dimensional array can be used, but the 
# number of components of one point (here: point of pos)
# must be equal to the number of components of the second point 
# (here: center point of center). 
# This is checked in the first line of the function.


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
def assign_closest_cluster(pos, centr, dist):
  # *** method .shape ***
  # here is a check if the amount of components of one point 
  # (here: point of pos) are equal to the amount of components 
  # of the second point (here: center point of center). 
  # The method .shape returns the dimension of an object, 
  # as (a,b), where a is number of array elements in axis0 direction 
  # (here: e.g. point of pos, the amount of centers specified),
  # and b is the amount of array elements in axis1 direction 
  # (here: the amount of components of one point). 
  # The values of pos.shape a,b can be accessed with 
  #   a = centr.shape[0] and b = centr.shape[1]. 

  # If the amount of components of one point (the point of pos) 
  # is not equal to the amount of components of the second point 
  # (the center point), the closest cluster cannot be assigned. 
  # In this case, the algorithm warns the user by printing out an 
  # error message,that shape[1] of pos and centr are not equal, 
  # i.e. the amount of array elements in axis1 direction are 
  # not equal. Since the program may crash without equal 
  # shape[1] of pos and center, the algorithm should not execute. 
  if(centr.shape[1] != pos.shape[1]):
    print("error in \"def closest_cluster(pos, centr)\" shape[1] \
    of pos and centr incompatible")

    # *** "error return" *** 
    # 1. Return exits the function. This return is aligned with
    # the error message inside the if, to induce function exit if 
    # the shape[1] of pos and centr are not equal. 
    # 2. The structure of this "error return" has to match the 
    # structure of the "normal" function return. 
    # assign_closest_cluster() returns TWO objects: 
    # closest_cluster and dist_point_cluster. Hence, also 
    # "error return" has to return TWO objects. 
    # These can be chosen arbitrary, but should indicate that 
    # an error has happened. In case of error, executing 
    # "print(closest_cluster)" will here output: "-1". 
    # An intact closest_cluster should contain the index of 
    # centr which corresponds to each point 
    # (here: the center point of centr that is closest 
    # to each point of pos). "-1" is not an index of the 
    # centr array. Hence, something is 
    # wrong with the execution of this function. 
    return -1, -1

  # *** closest_cluster: shape and content *** 
  # The new array closest_cluster should contain at 
  # each index the integer number of the center point 
  # in centr that corresponds to the point of pos
  # at the same index. Then, writing 
  # "print(closest_cluster[index])" with index 
  # replaced by the index number i, will output 
  # the index of centr that the data point
  # in pos with index i got assigned to.
  # Index must be integer, therefore the object type 
  # is casted as np.int32. 
  # closest_cluster is an array filled with zeros, 
  # containing the same amount of array elements in axis0 
  # direction as the array pos (containing the points of the set). 
  # Thus, the amount of indices of closest_cluster is 
  # equal to the amount of indices of pos.
  closest_cluster = np.int32(np.zeros([pos.shape[0]]))

  # *** dist_point_cluster: shape and content ***
  # The new array dist_point_cluster should contain at each index 
  # the computed distance of the point in pos at that index 
  # to its CLOSEST center point. The following algorithm will 
  # find out, which center is the closest for each data point, 
  # and store this result in closest_cluster. Then, writing 
  # "print(dist_point_cluster[index])" with index replaced by the 
  # index number i, will output the distance between the point
  # in pos with index i for axis0 (pos[i,:]).
  # Distance is a float, therefore the object type is not casted 
  # since np.zeros() produces arrays filled with float per default. 
  dist_point_cluster = np.zeros([pos.shape[0]])

  # *** first while: counters i_pos and i_c ***
  # There are two nested WHILE loops. 
  # Two counters are needed to calculate the distance of each
  # point of the set pos to each center point in centr. 
  # 1. One counter i_pos iterates over the indices of pos along axis0.
  # 2. For each of these points of pos, one counter i_c is needed 
  # to iterate over all center points (one center after the other). 
  # Hence, there is an iteration inside an iteration. 

  # *** counter i_pos: instantiation ***
  # The counter for iteration over the points of the set pos 
  # is called i_pos. The counter for the iteration over the
  # centers is called i_c. Any variable has to be instantiated 
  # first before it can be used. Instantiation means defining 
  # variable by writing it down and assigning a value to it. 
  # The counters get instanciated with the value 0 because the 
  # counting should start at index 0.
  i_pos = 0
  # *** first WHILE: iteration over i_pos ***
  # The iteration starts from index i_pos, which is 0 in the 
  # first round, to the index pos.shape[0] (the amount of points 
  # of the set pos - 1). The "-1" is due to the ``zero indexing'' 
  # in python: indexing does not go from 1:end, but from 0:(end-1).
  # The ``:'' indicate a subsetting of all the values between start
  # and end. 
  while i_pos < pos.shape[0]:
    # *** inside WHILE: instantiation of counter i_c ***
    # the counter for iteration over the centers, i_c, gets 
    # instantiated INSIDE the first while loop. In this way, 
    # every time, the next point of the set is reached, 
    # the counter i_c over the centers is reset to 0. Then, 
    # the distance can be computed of the next point in pos 
    # to ALL the centers 
    # (and not only the centr.shape[0]th center).  
    i_c = 0
    #
    # *** ele1: input of distance function ***
    # ele1 will be the input for the distance function. 
    # The distance function takes (ele1, ele2) as argument,
    # where ele1 is the point of the set pos at index i_pos, 
    # with all components.
    ele1 = pos[i_pos,:]

    # *** distance_start: store distance ***
    # This steps computes the distance between the selected point
    # of the set pos at index i_pos, and the center point of 
    # centr, at index i_c, with all components. 
    # The output is distance_start: the distance of a point 
    # of the set pos at index i_pos to the center point of centr 
    # at index i_c.
    distance_start = dist(ele1, centr[i_c,:])
    #
    # *** closest_cluster: first cluster assignment *** 
    # Here, the first cluster (center point of centr) gets 
    # assigned to the point of the set pos. 
    # The value of closest_cluster at the index i_pos is set 
    # to be the index of the corresponding center point in centr. 
    # In this way, the index i_pos of pos gets assigned to the 
    # index i_c of the center. printing closest_cluster at position
    # i_pos will output the index i_c of the center point in centr 
    # that is closest to the point of the set pos at index i_pos. 
    closest_cluster[i_pos] = i_c
    #
    # *** dist_point_cluster: saving the distance ***
    # dist_point_cluster contains at index i_pos: distance_start, 
    # the distance of the point of a set to the center point. 
    # This array will be updated in the following while loop. 
    dist_point_cluster[i_pos] = distance_start
    #
    # *** counter i_c: UPDATE ***
    # The counter for center is set to the next integer number. 
    # Now the above steps are repeated inside a new while.
    # This enables to compare the already computed distance 
    # between a point of the set pos at index i_pos and the 
    # center point at index i_c, with the other distances between 
    # a point of the set pos at index i_pos and center point 
    # at index (i_c+1). Note that the value stored inside i_c
    # gets overwritten by the new value i_c+1. 
    i_c=i_c+1
    #
    # *** second WHILE: iteration over i_c ***
    # The iteration over the center points starts. 
    # It lasts until the index i_c is smaller than the amount 
    # of center points in axis0 direction (until 
    # all centers have been iterated over). 
    while i_c < centr.shape[0]:
      # *** REMARK about loop initialization ***
      # A WHILE loop has to be initialised first. 
      # Therefore, the following code line are double, both in
      # in the first while, and here inside the second while. 
      #
      # *** ele 1 ***
      # ele1 is the input for the distance function. 
      # The steps above repeat (compare with: l. 186 ele1)
      ele1 = pos[i_pos,:]
      #
      # *** dist_compare: saving the second distance for comparison ***
      # The new distance (between the point of the set pos, 
      # as stored in ele1, and the next center point of centr) 
      # gets stored in distance_compare, not in distance_start. 
      # This is for comparing the value in distance_compare 
      # with the value in distance_start. 
      distance_compare = dist(ele1, centr[i_c,:])
      #
      # *** IF: Comparison of distances, saving the smallest ***
      # Here is the comparison, the smallest distance is saved. 
      # If the new distance, as saved in distance_compare 
      # (the distance between the point of the set pos and 
      # the next center point in centr) is SMALLER than the 
      # start distance, as saved in distance_start the distance 
      # between the point of the set and the previous center 
      # in centr), the value in distance_start gets overwritten 
      # by the value in distance_compare.  
      if (distance_compare<distance_start) == True:
        distance_start = distance_compare 
        # *** closest_cluster: UPDATE ***
        # Here, the content of closest_cluster is updated. 
        # If the currently selected center point is closer to the 
        # point than the previous center point (as measured by 
        # a smaller distance), the value at index i_c of 
        # closest_cluster gets overwritten by the value at 
        # the next index i_c (remember: before starting this while loop, 
        # the counter i_c has been increased by 1). Axis1 is 
        # irrelevant, only axis0 is needed to make the selection, 
        # since one index of axis0 unequivocally corresponds to 
        # one point of the set pos.  
        closest_cluster[i_pos] = i_c
        #
        # *** dist_point_cluster: UPDATE ***
        # Here, the content of dist_point_cluster is updated. 
        # If the assignment of the closest center point to a point
        # of the set pos changes, also the corresponding distance 
        # between that center point and the point of pos has 
        # to be stored, in dist_point_cluster. 
        # Remember: the content of distance_start has been 
        # overwritten with the content of distance_compare. 
        # Thus, here the new smallest distance gets assigned.
        # Remember also: we are still inside the IF! 
        # Any distance stored in distance_start is, at the end, 
        # always the smallest distance between two points. 
        dist_point_cluster[i_pos] = distance_start

      # *** counter i_c: UPDATE ***
      # i_c is the counter for the second while, 
      # the iteration over the centers. The same procedure 
      # as described above will be done for the next 
      # center point in centr, always taking distance_start 
      # as the reference distance for comparison. 
      # 
      # *** end of the second WHILE ***
      # When the entering condition of this second WHILE 
      # is not met anymore, that i_c < amount of 
      # center points in centr, the while is not entered anymore. 
      # 1. The smallest distance has been found and stored 
      # in dist_point_cluster at the same index as 
      # the corresponding point in pos. 
      # 2. The corresponding center to the point of the set
      # has been found and stored in closest_cluster at
      # the same index as the corresponding point in pos. 
      i_c = i_c+1
    #
    # *** counter i_pos: UPDATE ***
    # When this statement is reached in execution, the 
    # second WHILE has iterated over all center points 
    # in centr for a point of the set pos. 
    # This counter increases the index i_pos by 1. 
    # In this way, the next point of the set is selected 
    # for computing its distance to all center points of centr.
    i_pos = i_pos+1
  #
  # *** end of the first WHILE ***
  # The algorithm terminates when i_pos is not < the 
  # amount of points of the set anymore. 
  # It means that every point has been already selected, 
  # and the closest center for that point determined. 
  # 
  # *** RETURNS of assign_closest_cluster ***
  # 1. The function assign_closest_cluster returns 
  # closest_cluster. The amount of elements inside 
  # this array is equal to the amount of points of the set. 
  # At closest_cluster[i_pos] is the index of 
  # the center point in centr, that has been assigned 
  # to the point of the set at pos[i_pos,:]. 
  # Thus, centr[closest_cluster[i_pos],:] gives 
  # the centr point with ALL components that got assigned 
  # to point of the set at index pos[i_pos,:].
  # 2. The function assign_closest_cluster returns 
  # dist_point_cluster: the (smallest) distance between 
  # point of the set at pos[i_pos,:] and the 
  # closest center point at centr[closest_cluster[i_pos],:]. 
  return closest_cluster, dist_point_cluster
# Function END assign_closest_cluster() *************************



# Function START update_centr() ***************************************

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

# *** WHY update centr? ***
# In a clustering, centers are assigned to points of a set not
# only once, but multiple times. After each iteration 
# of the clustering, the center points are computed newly
# as the arithmetic mean of all the points of the set that 
# have been assigned to that center point in the previous round 
# of clustering. update_centr() computes the new center 
# points of the centr array, for the next round of clustering. 
# It takes closest_cluster to find out which points of 
# the set pos should be taken for each arithmetic mean, 
# to compute the new center point as the arithmetic mean of 
# these points of the set, for which pos[closest_cluster == i].
# In this way, only the points are selected that have been 
# assigned to the center point at index i.  
def update_centr(pos, centr, closest_cluster):

  # *** iterator ***
  # This iterator is a counter. iterator is an array
  # containing all integer numbers from 0 to the amount 
  # of points of the set pos.
  iterator = range(centr.shape[0])
  #
  # *** FOR: select indices of closest_cluster ***
  # iteration over all elements in iterator. This serves 
  # as a counter, to select all the indices at which 
  # closest_cluster contains the value i.  
  for i in iterator:
    #
    # *** selector ***
    # selector is an array containing all the indices at 
    # which closest_cluster contains the value i 
    # (the cluster number i). This can be used to select
    # all points in pos that have been assigned to this cluster. 
    selector = (closest_cluster == i)
    # 
    # *** IF: check if center has been assigned ***
    # The arithmetic mean of points assigned to a certain 
    # center point can only be calculated if the center point 
    # has actually been assigned. 
    # It can happen that a center point is not assigned.
    # This IF condition checks if selector contains any values 
    # larger than 0 (if any center point indices of centr are inside). 
    # If the sum of all elements in selector
    # is > 0, then the center at index i has been assigned 
    # at least once.  
    if np.sum(selector) > 0:
      # 
      # *** cluster_points: points assigned to a certain center ***
      # cluster_points contains all points that have 
      # been assigned to the center point at index i in centr 
      # (all the points that have been assigned
      # to that cluster i). The iteration is done along axis0 and axis1, 
      # the components (in axis1) of each point of the set are 
      # selected with ":". 
      cluster_points = pos[selector,:]
      #
      # *** arithm_mean: new center point ***
      # The arithmetic mean of the points of the set assigned 
      # to center i is computed with np.mean. Axis = 0 indicates 
      # the mean should be computed for each component separately, 
      # along axis0 (the mean of the x components, of the points will 
      # be the x component of the new center point, and so on 
      # for all components). The array arithm_mean contains 
      # pos.shape[1] amount of elements. 
      arithm_mean = np.mean(cluster_points, axis = 0)
    #
    # *** ELSE ***
    # If selector does not contain any value larger than 0 
    # (no point of the set has been assigned to center i), 
    # the else is entered. 
    else:
      #
      # *** No point assigned to center i: fake the new center i ***
      # The arithmetic mean is faked as an empty array of 
      # the same amount of components as the normal 
      # arithmetic mean of pos. It is important to preserve the 
      # structure of the array centr of centr points, and especially 
      # the indices of the center points. Then, a particular center 
      # can be clearly located by its index. Deleting center points 
      # that have not been assigned, would decrease the amount of 
      # center points inside centr, thereby shifting indices.  
      arithm_mean = np.zeros(centr.shape[1])
      # 
      # *** 'nan' ***
      # The array is filled with 'nan' ('not a number') to 
      # indicate no assignment has happened, and to prevent 
      # the program from crashing. Type float is chosen 
      # since the further algorithm works with float. 
      arithm_mean = arithm_mean*float('nan')
    # 
    # *** centr: UPDATE *** 
    # The new center point at index (i,:) of centr is the 
    # computed arithmetic mean. The arithm_mean has been computed 
    # along axis0, so component-wise. Therefore the shape of centr 
    # and arithm_mean are the same. 
    centr[i,:] = arithm_mean
  #
  # *** End of the FOR ***
  # This algorithm proceeds for all elements in iterator, 
  # i.e. until all center points have been updated.
  #
  # *** RETURNS of centr_update ***
  # The output is centr, filled with new center points 
  # (that can also be nan). 
  return centr
# Function END update_centr() *************************


# Function START accuracy() ***************************************
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
#
# *** WHY calculate accuracy? ***
# The accuracy function sets a limit when to stop the 
# clustering. It is defined as the amount that all 
# center points change from one clustering round to 
# another (as measured by a distance function). It 
# computes the sum of individual changes from each 
# center point to its old center point. 
# Once the change is lower than a specified accuracy 
# eps, the algorithm should terminate. 
# centr_old is instantiated in the function kmeans(), 
# it is the array of center points of the previous round 
# of clustering. 
def accuracy(centr, centr_old, dist):
  # 
  # *** new_eps: instantiation ***
  # new_eps is instantiated here. The value should 
  # be set to 0 to avoid interference with the specified 
  # eps. 
  new_eps = 0
  # 
  # *** iterator ***
  # this iterator iterates over all center points in center. 
  iterator = range(centr.shape[0])
  # 
  # *** FOR: iterate over all center points ***
  for i in iterator:
    # 
    # *** dist_centr_old: distance between new and old centr point ***
    # The dist_centr_old contains 1 value, the distance between 
    # the old center point at index i in centr_old 
    # and the updated center point at index i in centr. This value 
    # is stored in new_eps below, and will be summed with the 
    # distances from the other rounds of the FOR iteration. 
    dist_centr_old = dist(centr_old[i,:], centr[i,:])
    #
    # *** IF: select only center points which are NOT 'nan' ***
    # This IF checks if dist_centr_old contains a distance. 
    # If a center point has not been assigned, dist_centr_old
    # will not contain a distance. but 'nan' (assigned in update_centr()).
    # To prevent crashing of the algorithm, this dist_centr_old 
    # will be ignored for computing the new accuracy new_eps. 
    # If dist_centr_old does NOT contain 'nan', the IF is entered. 
    if math.isnan(dist_centr_old) == False: 
      #
      # *** new_eps: UPDATE ***
      # The new accuracy is the sum of all (distances of each 
      # center point to its old center point). In each round of 
      # iteration inside the for, the accuracy value for the 
      # next center point (stored in dist_centr_old) is added 
      # (to new_eps to update new_eps). 
      new_eps = new_eps + dist_centr_old
  # *** END of the FOR ***
  # *** RETURNS of accuracy() ***
  # new_eps is output: the accuracy of the current round 
  # of clustering, as measured by the changes between 
  # current and previous center points.  
  return new_eps
# Function END accuracy() *************************


# Function START kmeans() ***************************************
# # This functions makes a kmeans clustering. 
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
#   center, respectively). When this accuracy is reached, 
#   clustering stops. 
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

# *** kmeans: kmeans clustering ***
# This functions makes a kmeans clustering, with 
# the mean as the criterium for finding new centers. 
# The arguments have been specified above. New is max_iter: 
# an integer number to specify the maximum times the while 
# loop should execute.
def kmeans(pos, centr, eps, max_iter, dist):
  #
  # *** counter m: count rounds of clustering ***
  # m is the counter for max_iter, it counts the round 
  # of clustering.
  m = 0
  #
  # *** new_eps: instantiation ***
  # eps is the user-specified accuracy when to stop the algorithm. 
  # In the first round, the variable new_eps must contain a value
  # larger than eps, otherwise the WHILE is not entered
  # and the clustering is not done. Inside the WHILE, new_eps
  # is updated. For simplification, the value of eps + 1 
  # has been chosen as the starting value for new_eps. 
  new_eps = eps + 1
  #
  # *** eps_track: save all new_eps accuracies ***
  # this list contains all values of new_eps after each cluster round. 
  eps_track = []
  #
  # *** centr_old: copy of centr for accuracy() ***
  # Here is the copy of centr, filled with zeros. During each
  # clustering round, it will be copied anew from centr. 
  centr_old = np.zeros(centr.shape)
  # the operations done on centr necessitate it to be float. 
  centr = centr.astype(float)
  #
  # *** WHILE: kmeans *** 
  # Here, the kmeans iteration starts. The continuation criterium 
  # is that the new accuracy new_eps is still LARGER than the specified 
  # accuracy eps, and the round of clustering m is still SMALLER 
  # than the specified number of rounds max_iter. If new_eps 
  # becomes smaller or equal than the specified accuracy, 
  # OR the number of rounds specified has been reached, 
  # the while is not entered anymore. 
  while new_eps > eps and m < max_iter:
    #
    # *** FUNCTION CALL: assign_closest_cluster() ***
    # Here is the function called that assigns to each point of 
    # the set pos the center point in centr which is closest to 
    # that point (which has smallest distance to that point). 
    closest_cluster, dist_point_cluster = assign_closest_cluster(pos, centr, dist)
    #
    # *** centr_old: copy centr ***
    # Here, the values of centr get copied into centr_old.
    centr_old[:,:] = centr[:,:]
    #
    # *** FUNCTION CALL: update_centr() ***
    # Here is the function called that updates the centr values. 
    centr = update_centr(pos, centr, closest_cluster)
    #
    # *** FUNCTION CALL: accuracy() ***
    # Here is the function called that computes the accuracy 
    # (the change in center points between centr and centr_old)
    new_eps = accuracy(centr, centr_old, dist)
    #
    # *** eps_track: UPDATE ***.
    # Here the list eps_track gets filled with the new accuracy value new_eps,
    # which is output from accuracy().
    eps_track.append(new_eps)
    #
    # *** counter m: UPDATE ***
    # The counter m that counts the iterations, is increased by one. 
    # The WHILE restarts from the beginning, checking first IF 
    # the conditions are still met, and then entering the WHILE or 
    # exiting the function.
    m = m + 1
    #print(new_eps)
  
    # *** END of the WHILE *** 

  # *** VISUAL OUTPUT ***
  # This kmeans function contains a visual output to check how the 
  # accuracy changes. It plots eps_track. 
  plt.figure(figsize = (35, 35))
  plt.subplot(2,2,1)

  # *** Plot design ***
  # marker "s" means square, ls = '' is no line. 
  plt.plot(eps_track, marker="s", ls = '')
  plt.title('Change of accuracy eps')
  plt.yscale('log')
  plt.xlabel('Number of clustering round')
  plt.ylabel('eps')

  # *** RETURNS of kmeans() ***
  # Finally, the function kmeans() returns array centr of 
  # center points after clustering, and the array closest_cluster,
  # containing the number (as index) of the center point that got 
  # assigned to each point of the set in pos 
  # (at the same index as in closest_cluster). 
  # Happy clustering! 
  return centr, closest_cluster
# Function END kmeans() *************************
