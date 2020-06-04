r""" This file contains mask functions. 

"""

import numpy as np
import clustering.distance as distance
r"""
    - input:
        - r: the radius of the circular mask to be created. 
    - output:
        - mask: 
"""

def circular_mask(r):
    r""" create an array filled with zeros of shape
    2r in axis0 direction by 2r in axis1 direction.
    r is the radius of the circular mask to be created. 
    mask is a square, not a circle. In the following while loop, 
    this square will be made circular by eliminating certain values. 
    """
    mask = np.zeros((2*r,2*r))
    r""" centr_index gives the center point of the circular mask. 
    A circular mask is constructed as a hypothetical circle. 
    The key mathematical property of the circle used here is: 
    all points on the border of the circle are equally distant 
    from the center of the circle. The center point 
    will be used as a reference to eliminate all the values 
    in the mask that are more distant than a length r 
    to the center (those points that lie outside of the 
    radius of the circle). 
    """
    centr_index = np.array([r,r])
    r"""
    The iteration contains two while loops. One over the elements 
    of mask in axis0 direction, and one over the elements of mask in 
    axis1 direction. In this way, every point (with its two components) 
    will be compared with the center of the circle. Those points that
    more distant than the radius r (distance determined by the 
    euclidean distance) will be set to 1. Those points that are within
    the radius r of the circular mask, will be set to 0. In this way, 
    the filling pattern of the np.array mask will resemble a hypothetical 
    circle, where the circle is made from the 1 values and the background
    from the 0 values. 
    """
    # set axis0 counter to 0, instantiation of counter
    i = 0
    # instantiate first loop: elements in axis0 direction
    while i < mask.shape[0]:
        # set axis1 counter to 0, instantiation of counter
        j = 0
        # instantiate first loop: elements in axis0 direction
        while j < mask.shape[1]:
            # compare distance between point and center with radius r
            # if larger then r, enter the statement
            if distance.dist_euclidean(centr_index, np.array([i, j])) > r:
                # fill mask at index i,j with value 1
                mask[i,j] = 1
            # go here if the if condition is not matched.
            else:
                # fill mask at index i,j with value 0
                mask[i,j] = 0
            # increase axis1 counter by 1 (go to next element in axis1 direction)
            j = j + 1
          # increase axis0 counter by 1 (go to next element in axis0 direction)
        i = i + 1
    # output: mask, a np.array filled with 0 in the shape of a circle, and 1 at the edges. 
    return mask