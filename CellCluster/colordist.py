import clustering.kmeans_detailed as kmeans
import numpy as np



def colordist(col1, col2):
  colordist_v = col1[2]*col2[2]*kmeans.dist_euclidean(col1[0:2], col2[0:2])
  return colordist_v
