# This file contains functions that manipulate images. 
# Content(functions):
#     - rotate_vet()
#     - crop_image()
#     - img_rescaling()


import numpy as np
import skimage.transform as transform
import matplotlib.pyplot as plt




def crop_image(img_input, final_dim1, final_dim2):
    r""" This function crops an image starting from 0,0. 
    - input: 
        - img_input: image to be cropped. 
        
        - final_dim1: int of img.shape[0] after cropping 
        (length in axis0 direction)

        - final_dim2: int of img.shape[1] after cropping 
        (length in axis1 direction)

    - output:
        - img_cropped: part of image with specified dimensions
    """
    img_cropped = img_input[0:final_dim1, 0:final_dim2]
    return img_cropped


def img_rescaling(img_input, channel, rescaling_factor):
    r"""
    This functions applies the following changes to an image: 
    - rescaling
    - extract one channel, the number specified by the user. 
    Rescaling speeds up the computation, the extracted channel is needed
    for kmeans. 
    - input: 
        - img_input: a np.array to be rescaled
        - channel: the channel values to be extracted
        - rescaling_factor: a float or int
    - output: 
        - img_np: the cropped 3D image
        - img_channel: the 2D np.array with the extracted channel
    """
    r"""make a copy of the picture with only the channel 'blue' values
The image is given as a 3 channel image. However, the relevant information
(position of nuclei) can be extracted using only the information from one
channel. This is because the objects to be identified are in the original 
picture colored with the same color. 
    """
    r""" preserve the original image img_np by making a copy. 
The copy is called img_channel_2 since it contains only the values of 
channel 2, the blue channel. The resulting image is of size axis0: img_np.shape[0],
of size axis1: img_np.shape[1], of size axis2: 1. The resulting image is a
grayscale image since it has 2 dimensions.
    """

    img_channel = np.copy(img_input[:,:,channel])
    r""" Transformation
Here, the image gets rescaled to reduce the number of pixel while 
preserving the relevant information. This is for testing purposes 
to speed up the execution. 
The skimage.transform.rescale() function takes as arguments the image,
an integer as scaling factor, and anti_aliasing: border straight (false) 
or border smoothed (true). The original data should be preserved, so
false is chosen. rescale() returns a np.array, with every pixel with a 
component between 0 and 1 (normalised to 1). 
To compare it with the original image by view, all values of the array 
get multiplied with 255 (the maximum value for a color channel). 
"""
    img_channel = np.uint8(255*transform.rescale(img_channel, rescaling_factor, anti_aliasing=False))
    r"""
Here, a 3D image is reconstructed. 
This has to be done because the subsequent functions have been written for 
a 3D input. The image should be the same size as img_channel_2, but with 
3 channels instead of 1. These additional channels should be filled with zeros. 
np.uint8 is necessary because np.zeros produces a float by default, but for 
further computation integer (intensity values are needed). 
"""
    img_np = np.uint8(np.zeros([img_channel.shape[0], img_channel.shape[1], 3]))
    r"""
Here, the blue channel (2) of the still zeroed reconstructed image is filled
with the blue channel values stored in img_channel_2. 
"""
    img_np[:,:,2] = img_channel
    return img_np, img_channel


def treshold_values(img_input_2D, img_original, treshold):
    r""" This functions removes all values below the 
    specified treshold by setting them to 0. 
    By this, a selection is made which pixel will be 
    considered for the kmeans. 
    - input:
        - img_input_2D: a 2D np.array containing the 
        channel of interest, as extracted from an original
        image. 
        - treshold: specified treshold 

    - output:
        - nuclei: np 2D array containing the set of points 
        to be fed into kmeans (here: all points corresponding
        to a nucleus.)

    WARNING: img_input needs to be a 2D np.array. 
    """

    #     Treshold and find the pixels corresponding to a nucleus. 
    # A tresholding is applied to select the region which contains 
    # relevant information about the nuclei. Background information 
    # is not relevant, so will be set to 0 (black). Everything above treshold 
    # is relevant and will be set to 255 (white). 
    # Nuclei will appear as white spots on black background. 
    # Note:
    # the treshold value has to be sufficiently close to the color of 
    # the nuclei. There will be a variation in intensity values, for ex
    # values between 255-200. The tresholding uniforms all these values
    # to obtain a black-and-white-image of only 0 and 255. 
    # This is to simplify the selection of the cells (as all pixels with 
    # intensity 255). 

    img_input_2D[img_input_2D < treshold] = 0
    #img_input_2D[img_input_2D >= treshold] = 255

    # prepare set of points pos
    # The input pos of kmeans() should contain the points of the image, 
    # but only the most relevant information to identify a nucleus.
    # The image has been channel_extracted, the average radius of the cell
    # estimated, the channel_extracted image tresholded to an image of two 
    # different intensity values: 0 and 255. 
    # Relevant for the kmeans clustering are the points corresponding to 
    # each nucleus. The background is irrelevant for the clustering. 
      
    # Fed to kmeans() will be pos, a np.array containing the indices that 
    # inside the resulting pos, the background indices will be neglected. 
    # Thus, the task is, obtain the indices of white values (255) in img_channel_2. 
      
    # nuclei_axis0 contains the axis0 indices, nuclei_axis1 the axis1 
    # indices. This is because np.where gives the indices separately for each dimension. 
    # One axis0 index and the corresponding axis1 index (in the same position 
    # in nuclei_axis0 and nuclei_axis1) determine an index of a white pixel (value 255). 
      
    # We need a vector that contains the intensity values of the reconstructed image img_np 
    # (the image that contains 3 channels, of which only the blue channel has values non-zero.
    # It contains all the points that belong to a nucleus.)

    nuclei_axis0, nuclei_axis1 = np.where(img_input_2D >= treshold)

    # nuclei_axis2 contains the intensity values in the channel of img_np 
    # at the specified index. The values are divided by 255.0 to normalise 
    # the value between 0 and 1, and make them float and easy to handle.
    nuclei_c = img_original[nuclei_axis0, nuclei_axis1, 2]/255.0

    # The final set of points pos is created by joining pos_axis0, 
    # pos_axis1 and pos_axis2 one after the other, into one sequence of numbers.  
    nuclei = np.concatenate((nuclei_axis0, nuclei_axis1, nuclei_c), axis = 0)

    # The final pos has to have the shape required for the kmeans algorithm.
    # pos.shape[0] should be the number of points, pos.shape[1] the number of components. 
    nuclei_reshaped = nuclei.reshape([3, len(nuclei_axis0)])
    # flip axis0 and axis1 (transposition)
    nuclei = np.transpose(nuclei_reshaped)

    return nuclei





# This function rotates points in space by an angle theta. 
def rotate_vet(theta,vet_i):
  vet_rotated = np.copy(vet_i)
  vet_rotated[:,0] = np.cos(theta) * vet_i[:,0] + np.sin(theta) * vet_i[:,1] 
  vet_rotated[:,1] = - np.sin(theta) * vet_i[:,0] + np.cos(theta) * vet_i[:,1] 
  return vet_rotated