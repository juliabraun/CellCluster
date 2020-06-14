
# This file contains functions that find the initial centers 
# for kmeans clustering. 
# content(functions):
#   - create(centers). Display found center point as red pixel. 


import IO.load_image as loader
import os
import numpy as np
import skimage.io as io
import image_processing.mask as mk
import matplotlib.pyplot as plt
import image_processing.image_properties as image_properties


def create_centers(img_input, r, color_treshold):
    r""" This function creates an array of initial centers for kmeans().
    It is based on finding next-closest pixel with highest intensity,
    and then wiping out all pixel values within a radius r. 

        - input: 
            - img_np: a np.array of the preprocessed image

            - r: the radius of the circular mask to be created. 
            It corresponds to the average radius of the nuclei, 
            as estimated by radius.estimate_radius()

            - color_treshold: an integer number between 0-255. 
            This sets the treshold which intensity values to include. 

        - output:
            - save_c_max: a np.array containing the initial centers.
    WARNING:
        - img_input has to be a grayscale image (2D np.array) with the objects of 
        interest having higher intensity values than the background !
    """

    # make a copy of the input image
    img_np = np.copy(img_input[:,:,2])

    # cast radius to int
    r = np.int32(r)

    # define the dimensions of extended image
    ext1 = img_np.shape[0]+2*r
    ext2 = img_np.shape[1]+2*r

    # create the extended image 
    img_ext = np.zeros((ext1, ext2))
    
    # indexing for copying all img_np pixels into img_ext
    left_index = (r,r)
    right_index = (img_ext.shape[0]-r, img_ext.shape[1]-r)
    
    # select axis0 and axis1 values of img_ext which are to be 
    # replaced with img_np values.
    img_ext[left_index[0]:right_index[0], left_index[1]:right_index[1]] = img_np
    #io.imshow(img_ext)
    #io.show()
    #print(img_ext)

     # define the circular mask of radius r.  
    mask = mk.circular_mask(r)

    
    # WHILE INSTANTIATION
    # This loop finds out the positions of intensity values maxcol 
    # in the image. maxcol is initially set to 255, but 
    # gets updated during the loop and will correspond to the maximum
    # intensity value found in the image. Then, all pixels will be 
    # selected with the same intensity value. 
    
    maxcol = 255

    # create an empty list to save the maximum intensity value corresponding 
    # to the center of a nucleus. 
    
    save_c_max = []

    while maxcol > color_treshold:
         # find maximum intensity value in img_ext.
        maxcol = np.amax(img_ext)

        # find position of maxcol value
        img_whitex, img_whitey = np.where(img_ext == maxcol)

        # select the first position with maximum intensity value
        first = (img_whitex[0], img_whitey[0])
        
        # specify indices where to apply the mask
        left_index = (first[0]-r, first[1]-r)
        right_index = (first[0]+r, first[1]+r)
        
        # create a squared subselection of the img_ext whose size is equal to mask
        submattochange = img_ext[left_index[0]:right_index[0], left_index[1]:right_index[1]]
       
        # apply the mask
        img_ext[left_index[0]:right_index[0], left_index[1]:right_index[1]] = np.multiply(submattochange,mask)
        
        # show the cells replaced by the mask
        # io.imshow(img_ext)
        # io.show()
        
        # save the values of position and intensity
        list_save = [first[0]-r, first[1]-r, maxcol]
        
        # put list_save list into save_c_max
        save_c_max.append(list_save)

    # cast save_c_max to int
    save_c_max = np.int32(np.array(save_c_max))

    i = 0
    while i < save_c_max.shape[0]:
        
        # This while iterates over all found center pixels of
        # the nuclei and replaces their color with red 
        # (channel 0, intensity 255). 
        
        img_input[save_c_max[i,0], save_c_max[i,1], 0] = 255
        i = i+1
    
    #r"""
    #Display image of the nuclei whose found center pixel 
    #is colored red. 
    #"""
    #plt.figure()
    #io.imshow(img_input)
    #io.show()
        
    return save_c_max # np.array that contains int of position and intensity of the centers


def random_centers(k,):
    r""" A function that computes random intial centers. 
    k has to be given. Not working yet. 
    """
    #centr = np.random.random((k, pos.shape[1]))
    return