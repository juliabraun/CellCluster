import numpy as np

def randomset(dilation_factor, translation_axis0, translation_axis1, number_components = 2):
    r""" This function creates clouds of points for clustering.  
    
    - input:
        - dilation_factor: a int or float that specifies the dilation/contraction of the random point set
        - translation_axis0: int or float specifying the shift in axis0 direction
        - translation_axis1: int or float specifying the shift in axis1 direction
        - number_components: int specifying number of elements in axis1 direction. Default is 2. 

    - output:
        - randomset_v: a np.array random set of random values, shifted in axis0 and axis1 direction,
        perhaps dilated/contracted. 

    - info: np.random.rand(d0, ..., dn) creates random values in a given shape (of n dimensions). 
        The output is a float between 0-1. 

    - WARNING: 
        The user has to specify number_components: the number of components of each point 
        of the set (in axis 1 direction). If not given, default is 2. 
        The amount of random points is defined as 1000 * number_components. 
    """

    # 0.5: shift the point range:  -0.5 to 0.5. Hypothetical middle point of the set is 0,0. 
    randomset_v = np.random.rand(number_components*1000, number_components) - 0.5
   
    # specify dilation/contraction factor
    randomset_v = dilation_factor*randomset_v

    # translate components of axis0
    randomset_v[:,0] = randomset_v[:,0] + translation_axis0

    # translate components of axis1
    randomset_v[:,1] = randomset_v[:,1] + translation_axis1
    return randomset_v
