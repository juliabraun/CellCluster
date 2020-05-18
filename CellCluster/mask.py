import numpy as np
import clustering.kmeans_detailed as kmeans


def circular_mask(r):
  mask = np.zeros((2*r,2*r))
  centr_index = np.array([r,r])

  i = 0
  while i < mask.shape[0]:
    j = 0
    while j < mask.shape[1]:
      if kmeans.dist_euclidean(centr_index, np.array([i, j])) > r:
        mask[i,j] = 1
      else:
        mask[i,j] = 0
      j = j + 1
    i = i + 1
  return mask
