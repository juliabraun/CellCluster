import numpy as np



# - this function calculates the manhattan distance between the 
#   two points (e.g. point of a set and center point). 
#           It returns a positive real number given two arrays. 
# - input: 
#     - ele1, ele2: numpy arrays of length pos.shape[1].
# - output:
#     - sum: the manhattan distance as sum of the distances of 
#     the point components: |x1 - x2| + |y1 - y2|+ ... + |n1-n2|.

def dist_manhattan(ele1, ele2):
  # ele1 is a point (here: of the set pos), ele2 is a second point 
  #   (here: the center).
  # np.abs returns the absolute value (negative values become 
  #   positive). 
  # dist1 therefore is a array with index-wise differences, 
  #   it contains: 
  #   x1-x2 at index 0,  y1-y2 at index 1, ..., n1-n2 at 
  #     index n-1. 
  # The letter indicates the component (x,y,...n), 
  # the number indicates the point (1: point of a set, 2: center point).
  dist1 = np.abs(ele1 - ele2)
  # the manhattan distance is defined as the sum of the 
  # components of dist1: |x1 - x2| + |y1 - y2|+ ... + |n1-n2|. 
  # np.sum makes index-wise sum over the values of all indices. 
  sum = np.sum(dist1)
  # the output is sum: the manhattan distance between one point 
  #   of the dataset (here: point of a set) 
  #   and the second point (here: center point). 
  return sum



# - this function calculates the euclidean distance of the 
#     positions of two points (e.g. point of a set and center point). 
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
  #   the number indicates the point (1: point of a set, 2: center point)
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
  # It gives a number representing the sum of the squared 
  # differences between the same component of two different 
  # points. 
  sum = np.sum(squared)
  # np.sqrt returns the squareroot of the input value, 
  # which is the euclidean distance between one point 
  # (here: of a set) and the second point
  # (here: center point).
  root = np.sqrt(sum)
  return root



def dist_colorweight(ele1, ele2):
    r"""
    This function computes the weighted distance between two points 
    using the color intensity as weight. 

    - input: 
        - ele1, ele2: numpy arrays of length pos.shape[1]. 
        The syntax of col1 is: index 0: position in axis0 direction,
        index 1: position in axis1 direction, 
        index 2: color intensity value. ele1[2] gives the intensity value. 
        Note: this algorithm works with binary images: the color of a 
        pixel is expressed as a number between 0-256. If the colors do not appear 
        as black and white in the image, this is because a colormap is applied. 

    - output: 
        - dist_colorweight_v: the distance value between one point
       of the dataset (here: point of a set) and the second point 
       (here: center point), taking into account the color intensity value

    The function multiplies the intensity values of both points by their euclidean
    distance. Thereby, points that have a low intensity value (corresponding to the 
    intensity of interest) receive a lower distance dist_colorweight_v, 
    and therefore are considered more important for belonging to a center point. 
    """
   
    dist_colorweight_v = ele1[2]*ele2[2]*dist_euclidean(ele1[0:2], ele2[0:2])
    return dist_colorweight_v
