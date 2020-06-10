import numpy as np

def randomset(dilation_factor, translation_axis0, translation_axis1, number_components = 2):
    r""" This function creates clouds of points for clustering. 
    
    - input:
        - dilation_factor: a int or float that specifies the dilation/contraction of the random point set
        - translation_axis0: int or float specifying the shift in axis0 direction
        - translation_axis1: int or float specifying the shift in axis1 direction
        - number_components: int specifying number of elements in axis1 direction. Default is 2. 

    - output:
        - randomset_v: a np.array as a random set of points, shifted in axis0 and axis1 direction,
        perhaps dilated/contracted. 

        np.random.rand(d0, ..., dn) creates random values in a given shape (of n dimensions). 
        The output is a float between 0-1. 
        randomset_v is a set of random values stored as a np.array. 
        The user has to specify number_components: the number of components
        of each point of the set (in axis 1 direction). The amount of random points 
        is defined as 1000 * number_components. 
        
        The points in randomset_v are shifted by - 0.5 to rescale the output symmetrically
        (float between - 0.5 and 0.5). In this way, the hypothetical middlepoint of the set is at 
        0,0. The set of points can be translated to a point of interest by imagining it as 
        translating the middlepoint of the set. 
    """

    # 0,5: shift the point range to  -0.5 to 0.5. Hypothetical middle point of the set is 0,0. 
    randomset_v = np.random.rand(number_components*1000, number_components) - 0.5

    r"""
    # the set of points gets multiplied by expansion_factor: to expand or contract the point cloud. 
    # The absolute values increase or decrease. 
    """
    randomset_v = dilation_factor*randomset_v

    r""" translate components of axis0. All the elements of axis0 and the first element of axis1 get 
     replaced by the same value plus a translation_axis0. In this way, the set of points can be 
     shifted in axis0 direction. 
    """
    randomset_v[:,0] = randomset_v[:,0] + translation_axis0

    # translate components of axis1
    randomset_v[:,1] = randomset_v[:,1] + translation_axis1
    # return randomset_v: a random set of points shifted by translation_axis0 in axis0 direction,
    # by translation_axis1 in axis 1 direction, and dilated/contracted by dilation_factor
    return randomset_v
