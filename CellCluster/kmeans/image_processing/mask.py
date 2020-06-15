r""" This file contains mask functions. 

"""

import numpy as np
import kmeans.clustering.distance as distance



def circular_mask(r):
    # create 
    mask = np.zeros((2*r,2*r))
    centr_index = np.array([r,r])

    i = 0
    while i < mask.shape[0]:
        j = 0
        while j < mask.shape[1]:
            if distance.dist_euclidean(centr_index, np.array([i, j])) > r:
                mask[i,j] = 1
            else:
                mask[i,j] = 0
            j = j + 1
        i = i + 1
    return mask


