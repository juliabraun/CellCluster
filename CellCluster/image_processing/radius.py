r"""
This file contains functions to find out the radius of the cells. 
content(files):
    - autocorr()
    - estimate_radius()
"""
import numpy as np
import IO.load_image as loader
import skimage.io as io


def autocorr(img_channel_2):
    """
    This makes autocorrelation
    first fourier transform in 2D, then inverse fourier transform 2D
    then shift 
    """
    fourier = np.fft.fft2(img_channel_2)
    autocorr_v = np.fft.ifft2(np.power(np.abs(fourier), 2))
    shift = np.fft.fftshift(np.abs(autocorr_v))
    
    return shift, autocorr_v

def estimate_radius(img_channel_2):
    """
    this function computes the average radius of the cells, 
    based on half_maximum at half width
    All values of the 3D function at point half maximum half width
    Values below the point are excluded
    result is a sharpened contour
    count with lenght of np.where 

    """
    shift, autocorr_v = autocorr(img_channel_2)
    maximum = np.max(shift)
    half_maximum = maximum/2
    find = np.where(shift < half_maximum) # reduce redundancy in this code !!!!!!!!!!
    shift[find] = shift[find]*0
    circlex, circley = np.where(shift>0)
    count = circlex.shape[0]
    r = np.sqrt(count/np.pi)
    print("The estimated average radius of the nuclei is " + str(r) + " pixel.")

    return r

    #io.imshow(shift)
    #io.show()