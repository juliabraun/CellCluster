import numpy as np
import IO.load_image as loader
import skimage.io as io


def autocorr(img_blue):
    """
    This makes autocorrelation
    first fourier transform in 2D, then inverse fourier transform 2D
    then shift 
    """
    fourier = np.fft.fft2(img_blue)
    autocorr_v = np.fft.ifft2(np.power(np.abs(fourier), 2))
    shift = np.fft.fftshift(np.abs(autocorr_v))
    
    return shift, autocorr_v

def estimate_radius(img_blue):
    """
    this function computes the average radius of the cells, 
    based on half_maximum at half width
    All values of the 3D function at point half maximum half width
    Values below the point are excluded
    result is a sharpened contour
    count with lenght of np.where 

    """
    shift, autocorr_v = autocorr(img_blue)
    maximum = np.max(shift)
    half_maximum = maximum/2
    find = np.where(shift < half_maximum) # reduce redundancy in this code !!!!!!!!!!
    shift[find] = shift[find]*0
    circlex, circley = np.where(shift>0)
    count = circlex.shape[0]
    r = np.sqrt(count/np.pi)
    print("HERE" + str(r))

    io.imshow(shift)
    io.show()