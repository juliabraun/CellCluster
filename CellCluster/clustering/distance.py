import numpy as np



# - This function calculates the manhattan distance between the 
#   positions of two points 
#   (e.g. point of a set and center point). 
#           It returns a positive real number given two arrays. 
# - input: 
#     - ele1, ele2: numpy arrays of length pos.shape[1].
# - output:
#     - sum: the manhattan distance as sum of the distances of 
#     the point components: |x1 - x2| + |y1 - y2|+ ... + |n1-n2|.

def dist_manhattan(ele1, ele2): 
  # make absolute value
  dist1 = np.abs(ele1 - ele2) 
  # sum all components (x with x, y with y ...)
  dist1 = np.sum(dist1)
  # return manhattan distance
  return dist1 



# - This function calculates the euclidean distance of the 
#     positions of two points (e.g. point of a set and center point). 
#     It returns a positive real number given two arrays. 
# - input: 
#     - ele1, ele2: numpy arrays of length pos.shape[1].
# - output:
#     - sum: the euclidean distance as squareroot of the 
#     sum of the squared differences between the components of 
#     two points: sqrt((|x1 - x2|)^2 + (|y1 - y2|)^2 + ... + (|n1 - n2|)^2).

def dist_euclidean(ele1, ele2):
  # make absolute value
  dist2 = np.abs(ele1 - ele2) 
  # (x1-x2)^2 and (y1-y2)^2
  squared = np.power(dist2, 2) 
  # sum the squared components (x^2 + y^6)
  dist2 = np.sum(squared) 
  # root
  root = np.sqrt(dist2) 
  # pythagorean hypothenusis
  return root 


# This function computes the weighted distance using the color intensity as weight. 
# For our project this is advantageous. More comments will follow.
def colordist(col1, col2):
  colordist_v = col1[2]*col2[2]*dist_euclidean(col1[0:2], col2[0:2])
  return colordist_v
