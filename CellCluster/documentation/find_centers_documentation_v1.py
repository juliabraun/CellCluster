r"""
This file contains functions that find the initial centers 
for kmeans clustering. 
"""

import IO.load_image as loader
import os
import numpy as np
import skimage.io as io
import mask as mk
import matplotlib.pyplot as plt


def create_centers(img_input, r, color_treshold):
    r""" This function creates an array of initial centers for kmeans().
    It is based on finding next-closest pixel with highest intensity,
    and then wiping out all pixel values within a radius r. 
        - input: img_np: a np.array of the preprocessed image
            - r: the radius of the circular mask to be created. 
        - r: the average radius of the nuclei, as estimated by 
        the radius.estimate_radius()
        - color_treshold: an integer number between 0-255. 
        This sets the treshold which intensity values to include. 
        - output:
            - save_c_max: a np.array containing the initial centers.
    WARNING:
        - img_np has to be a grayscale image (2D np.array) with the objects of 
        interest having higher intensity values than the background !
    """
    r""" 
    For the algorithm to be not collapsing, 
    it must be possible to select a mask area at the border. 
    The image is therefore extended by r pixels at each side.
    This code makes a copy of the image that contains 
    only the channel 2 values. 
    """
    img_np = np.copy(img_input[:,:,2])

    r"""
    r will be used to define the shape of the extended image and 
    therefore must be cast to integer. 
    """
    r = np.int32(r)

    r"""
    The two dimensions of the extended image are defined. 
    The dimensions of img_np are extended by 2r in both directions. 
    """
    ext1 = img_np.shape[0]+2*r
    ext2 = img_np.shape[1]+2*r

    # create the extended image
    img_ext = np.zeros((ext1, ext2))
    r""" Choose the indices of the extended, still empty img_ext, at which 
    the old image img_np should be inserted. The left_index is not 0,0 because
    there should be kept a border of r around the image. The closest possible
    point is r,r. There is r in axis0 direction and r in axis1 direction. 

    The right_index corresponds to the new image extension in axis0 direction - r,
    and the extension in axis1 directon - r. 

    """
    left_index = (r,r)
    right_index = (img_ext.shape[0]-r, img_ext.shape[1]-r)

    r"""
    The zeroes at the indices positions get replaced with the values from img_np. 
    The operation selects a rectangle whose side lenghts are specified by the indices. 
    """
    img_ext[left_index[0]:right_index[0], left_index[1]:right_index[1]] = img_np
    #io.imshow(img_ext)
    #io.show()
    #print(img_ext)


     # define the circular mask of radius r.  
    mask = mk.circular_mask(r)

    r"""
    WHILE INSTANTIATION
    This loop finds out the positions of intensity values maxcol 
    in the image. maxcol is initially set to 255, but 
    gets updated during the loop and will correspond to the maximum
    intensity value found in the image. Then, all pixels will be 
    selected with the same intensity value. 
    """
    maxcol = 255

    r""" create an empty list to save the maximum intensity value corresponding 
    to the center of a nucleus. 
    """
    save_c_max = []

    r"""
    Condition for this while is: the maximum intensity value found is still larger
    than the set intensity treshold. 
    """
    while maxcol > color_treshold:
        r"""
        maxcol is the np.array of all maximum intensity value 
        """
        maxcol = np.amax(img_ext)
        r"""
        Two arrays containing the indices in axis0 and axis1 direction of img,
        which point to the intensity value maxcol (255). 
        """    
        img_whitex, img_whitey = np.where(img_ext == maxcol)
    
        r"""
        Here, the indexing starts. 
        A selection of the image has to be made. The image selection is a circle 
        with the center point being the intensity value at indices 0,0 of the 
        img_whitex. To make the circle selection, the mask is applied. 
        The mask has to be applied by multiplying its values with the appropiate
        selection of the image. The resulting values are then replaced in the image. 
        For this, the correct indices of starting and ending have to be specied. 
        The selection mask is a square, so indices have to be specified for 
        a square of the image. 
        """
        r""" This variable contains the first values of the two arrays containing
        the indices in axis0 and axis1 direction. This corresponds to the first 
        pixel of maximum intensity in the image. 
        """
        first = (img_whitex[0], img_whitey[0])
        r""" The index spans from the upper left corner to the lower right corner
        of the squared mask. The new left_index are made by subtracting the radius r
        in axis0 and axis1 direction. The new right_index are made by adding the radius r
        in axis0 and axis1 direction.

        """
        left_index = (first[0]-r, first[1]-r)
        right_index = (first[0]+r, first[1]+r)
        r"""
        submattochange is a subset of the image array in which the squared around the 
        found maximum intensity value is stored. axis0 values contain all from the 
        left index to the right index (of axis0). axis1 values contain all from the
        left index to the right index (of axis1). 
        """
        submattochange = img_ext[left_index[0]:right_index[0], left_index[1]:right_index[1]]
        r"""
        The squared selection is replaced with the mask values. The image intensity values 
        are zeroed out. The selection is: all axis0 indices from left_index to right_index; 
        and all axis1 indices from left_index to right_index. 

        """
        img_ext[left_index[0]:right_index[0], left_index[1]:right_index[1]] = np.multiply(submattochange,mask)
        #io.imshow(img_ext)
        #io.show()
        r""" This list saves the indices of the found pixel of highest intensity,
        which corresponds to the center of the nucleus; and the intensity value. 
        We are operating on an extended image (+r in every direction), so the
        indices in the original image img_np are those in img_ext -r. 
        """
        list_save = [first[0]-r, first[1]-r, maxcol]
        r"""
        After the while loop, the saved points (corresponding to the center point
        of the nucleus) are formatted as int np.array.
        """
        save_c_max.append(list_save)

    r"""
    save_c_max will be used for pixel indexing below.
    Therefore it must be cast to int. 
    """
    save_c_max = np.int32(np.array(save_c_max))

    i = 0
    while i < save_c_max.shape[0]:
        r"""
        This while iterates over all found center pixels of
        the nuclei and replaces their color with red 
        (channel 0, intensity 255). 
        """
        img_input[save_c_max[i,0], save_c_max[i,1], 0] = 255
        i = i+1
    
    r"""
    Display image of the nuclei whose found center pixel 
    is colored red. 
    """
    plt.figure()
    io.imshow(img_input)
    io.show()
        
    return save_c_max



def random_centers(k,):
    r""" A function that computes random intial centers. 
    k has to be given. Not working yet. 
    """
    #centr = np.random.random((k, pos.shape[1]))
    return