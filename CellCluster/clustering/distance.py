import numpy as np



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



def colordist(col1, col2):
  colordist_v = col1[2]*col2[2]*dist_euclidean(col1[0:2], col2[0:2])
  return colordist_v
