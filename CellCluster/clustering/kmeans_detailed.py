# ---- INFO -----
# This is the version with concise comments. For detailed comments,
# see file kmeans_detailed_documentation_v1.py.
# ---------------

import numpy as np
import matplotlib.pyplot as plt
import math


#HERE Functions ________________________


def assign_closest_cluster(pos, centr, dist):
    r""" 
    This function calculates the closest cluster for each point 
    inside a set of points. 
    center points of the clusters must be provided.

    - input: 
        - pos: numpy.array as matrix containing the set of points. 
        pos.shape[0] must be the amount of points. 
        pos.shape[1] is the amount of components defining each point.
        
        - centr: numpy.array as matrix containing the set of centers 
        of the clusters. 
        centr.shape[0] must be the number of centers. 
        centr.shape[1] is the amount of components defining each center.

        - dist: a function that returns a positive real number given 
        two points. It will be used for calculating the distance 
        between points. In this function, the points are numpy arrays 
        of length pos.shape[1].

    - output: 
        - closest_cluster: a numpy.array of length pos.shape[0] 
        that contains the cluster number corresponding to each point 
        in pos.

        - dist_point_cluster: a numpy.array of length pos.shape[0] 
        that contains the distance dist of each point in pos to 
        its closest center.

    WARNING: 
        - centr.shape[1] must be equal to pos.shape[1]. 
        - The sequence and the amount of components of points must
        be equal in centr and pos. Any-dimensional array can be used.
        Compatibility is checked in the first line of the function.
    """

  # check for same amount of components
    if(centr.shape[1] != pos.shape[1]): 
        # if not matching, return error
        print("error in \"def closest_cluster(pos, centr)\" shape[1] \
        of pos and centr incompatible") 
        return -1, -1

    # ------------------------------------------
    # memory area definition 
    # storage of cluster assignment
    closest_cluster = np.int32(np.zeros([pos.shape[0]])) 
    # storage of distance
    dist_point_cluster = np.zeros([pos.shape[0]]) 
    # ------------------------------------------
    # initialise counter for iteration over the indices of pos
    i_pos = 0 

    # loop over all indices of pos
    while i_pos < pos.shape[0]: 

      # -------------------------------------------------
      # Initializing cycle over centr points, please read comment at HERE below
      # initialise counter for iteration
      i_c = 0 
      # input for distance function
      ele1 = pos[i_pos,:] 
      # compute distance between point and center
      distance_start = dist(ele1, centr[i_c,:])  
      # save assigned cluster at same index as point of pos
      closest_cluster[i_pos] = i_c 
      # save distance
      dist_point_cluster[i_pos] = distance_start 
      # increase counter
      i_c=i_c+1 
      # -------------------------------------------------

      # loop over all indices of centr
      while i_c < centr.shape[0]: 

          # ----------- HERE ------------
          #  If the distance between the point and the current
          # center is smaller than between the previous center, 
          # this center is a good candidate to be the closest. 
          # Save the info by overwriting distance_start !
          ele1 = pos[i_pos,:]
          # compute distance between point and next center
          distance_compare = dist(ele1, centr[i_c,:]) 
          # check which distance is smaller
          if (distance_compare<distance_start) == True: 
              # save smallest distance in distance_start
              distance_start = distance_compare 
              # save assigned cluster at same index as point of pos
              closest_cluster[i_pos] = i_c  
              # save distance
              dist_point_cluster[i_pos] = distance_start 
              # ---------------------------

          # next center
          i_c = i_c+1 
      # next point of pos
      i_pos = i_pos+1 
    # After this function, all point-center 
    # comparisons have been made, and the closest center for each point extracted.
    return closest_cluster, dist_point_cluster 



def update_centr(pos, centr, closest_cluster):
    r""" This function updates the center list centr

    - input:
        - pos: numpy.array as matrix containing the set of points. 
        pos.shape[0] must be the amount of points. 
        pos.shape[1] is the amount of components of each point.

        - centr: numpy.array as matrix containing the set of centers 
        of the clusters. 
        centr.shape[0] must be the amount of centers. 
        centr.shape[1] is the amount of components defining each center.

        - closest_cluster: a numpy.array of length pos.shape[0] that 
        contains the cluster number corresponding to each point in pos.

    -  output:
        - centr: numpy.array as matrix containing the updated set of centers
       of the clusters.
       Centr is now updated and ready for the next round of clustering.
    """

    # iteration over all centers
    iterator = range(centr.shape[0])  
    for i in iterator: 
        # create array with index of pos whose closest center is i
        selector = (closest_cluster == i)  
   
        # check if center i has been assigned at least once.
        if np.sum(selector) > 0:  
            # select points of a cluster
            cluster_points = pos[selector,:] 
            # compute new center
            arithm_mean = np.mean(cluster_points, axis = 0) 
      
        # if cluster not assigned, fake center
        else: 
            arithm_mean = np.zeros(centr.shape[1]) 
            arithm_mean = arithm_mean*float('nan')   
        # set new center as mean
        centr[i,:] = arithm_mean 
    # updated centr array
    return centr 


def accuracy(centr, centr_old, dist):
    r"""This function calculates the sum of (distances between 
    each old center point and the new center point, respectively).

    - input: 
        - centr: numpy.array as matrix containing the set of 
        center points of the clusters. 
        centr.shape[0] must be the number of centers. 
        centr.shape[1] is the number of components defining each center.

        - centr_old: same specification of centr, contains the 
        centr points of the previous clustering round.

        - dist: a function that returns a positive real number 
        given two arrays. 
        It will be used for calculating the distance between two points. 
        In this function, the points are numpy arrays of length pos.shape[1].

    - output:
        - new_eps: A float representing the sum of (distances between 
        each old center point and the new center point, respectively).
    """

    # storage for computed accuracy
    new_eps = 0 
    # iterate over all centers
    iterator = range(centr.shape[0]) 
    for i in iterator:
        # save distance between old and new center of each cluster
        dist_centr_old = dist(centr_old[i,:], centr[i,:]) 
        # exclude centers which are nan
        if math.isnan(dist_centr_old) == False: 
            # sum all accuracies
            new_eps = new_eps + dist_centr_old 
    # total center change from previous to current clustering round
    return new_eps  



def kmeans(pos, centr, eps, max_iter, dist): 
    r""" This functions makes a kmeans clustering.
    
    - input: 
        - pos: numpy.array as matrix containing the set of points. 
        pos.shape[0] must be the amount of points. 
        pos.shape[1] is the amount of components defining each point.

        - centr: numpy.array as matrix containing the set 
        of centers of the clusters. 
        centr.shape[0] must be the amount of centers. 
        centr.shape[1] is the amount of components defining each center.

        - eps: A float representing the required sum of (distances between 
        each old center and the new center, respectively).
        When this accuracy is reached, clustering stops. 

        - max_iter: an integer number specifying the maximum number of 
        clustering iterations.

        - dist: a function that returns a positive real number given two arrays. 
        It will be used for calculating the distance between two points. 
        In this function, the points are numpy arrays of length pos.shape[1].

    - output:
        - centr: numpy.array as matrix containing the set of centers 
        of the clusters. 
        centr.shape[0] must be the amount of centers. 
        centr.shape[1] is the amount of components defining each center. 

        - closest_cluster: a numpy.array of length pos.shape[0] that 
        contains the cluster number corresponding to each point in pos.
    """

    # ----------------------------
    # Memory area definition -----
    # counter of iterations
    m = 0 
    # initialise accuracy check, please read at HERE below
    new_eps = eps + 1 
    # save all eps after each round
    eps_track = [] 
    # save the centers of previous round
    centr_old = np.zeros(centr.shape) 
    centr = centr.astype(float)
    # ---------------------------

    # ----- HERE -------
    # to enter the while, new_eps needs to be larger than eps 
    # ------------------
    # two criteria: accuracy and round number
    while new_eps > eps and m < max_iter: 
        closest_cluster, dist_point_cluster = assign_closest_cluster(pos, centr, dist)
        # copy centers
        centr_old[:,:] = centr[:,:] 
        # set new centers
        centr = update_centr(pos, centr, closest_cluster) 

        # new accuracy
        new_eps = accuracy(centr, centr_old, dist) 
        # save new accuracy
        eps_track.append(new_eps) 
   
        # next round of clustering
        m = m + 1 
        #print(new_eps)

    # -----------------------------
    # VISUAL OUTPUT ---------------
    plt.figure(figsize = (35, 35))
    plt.subplot(2,2,1) 
    plt.plot(eps_track, marker="s", ls = '')  # marker s: square; ls '': no line
    plt.title('Change of accuracy eps')
    plt.yscale('log')
    plt.xlabel('Number of clustering round')
    plt.ylabel('eps')

    # Happy clustering! 
    # centers after clustering, points assigned 
    return centr, closest_cluster 
