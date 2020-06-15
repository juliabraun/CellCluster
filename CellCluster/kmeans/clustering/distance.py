import numpy as np





def dist_manhattan(ele1, ele2): 
  r""" the r indicates raw docstring, for any backslashes
    This function calculates the manhattan distance between the 
    positions of two points (e.g. point of a set and center point).
    It returns a positive real number given two arrays. 

    - input: 
        - ele1, ele2: numpy arrays of length pos.shape[1].

    - output:
        - sum: the manhattan distance as sum of the distances of 
        the point components: |x1 - x2| + |y1 - y2|+ ... + |n1-n2|.
  """
  # make absolute value
  dist1 = np.abs(ele1 - ele2) 
  # sum all components (x with x, y with y ...)
  dist1 = np.sum(dist1)
  # return manhattan distance
  return dist1 




def dist_euclidean(ele1, ele2):
  r"""
    - This function calculates the euclidean distance of the 
    positions of two points (e.g. point of a set and center point). 
    It returns a positive real number given two arrays. 

    - input: 
        - ele1, ele2: numpy arrays of length pos.shape[1].

    - output:
        - sum: the euclidean distance as squareroot of the 
        sum of the squared differences between the components of 
        two points: sqrt((|x1 - x2|)^2 + (|y1 - y2|)^2 + ... + (|n1 - n2|)^2).
  """
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


def dist_colorweight(ele1, ele2):
    r"""
    This function computes the weighted distance between two points 
    using the color intensity as weight. 

    - input: 
        - ele1, ele2: numpy arrays of length pos.shape[1]. 
        index 0: position in axis0 direction,
        index 1: position in axis1 direction, 
        index 2: color intensity value.

    - output: 
        - dist_colorweight_v: the distance value between one point
       of the dataset (here: point of a set) and the second point 
       (here: center point), taking into account the color intensity value

    Note: this algorithm works with binary images: the color of a 
    pixel is expressed as a number between 0-255.

    The function multiplies the intensity values of both points by their euclidean
    distance. Thereby, points that have a higher intensity value (corresponding to the 
    intensity of interest) receive a higher distance dist_colorweight_v, 
    and therefore are considered more important for belonging to a center point. ????
    """
    dist_colorweight_v = ele1[2]*ele2[2]*dist_euclidean(ele1[0:2], ele2[0:2])
    return dist_colorweight_v
