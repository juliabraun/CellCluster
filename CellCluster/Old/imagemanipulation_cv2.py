r"""
This file is the image preparation for kmeans, using predefined functions from cv2. 
The self-written image preparation functions are in image_loading_v1.py and use skimage. 
"""

import cv2
import numpy as np

#loads a single image. 
  #Input is filename: str of image location. Output is the image loaded from the file. 
def load_singleimage_cv2(filename):
  im_original = cv2.imread(filename)
  return im_original


#image_preparation prepares the loaded image for application of kmeans onto it (3D image: width, height, depth: 3 RGB values)
# kmeans takes as input 2D array np.float32 (2D image: number of pixel, 3 RGB values) 
  #input is im: image (2D array), output is im_vet: the reshaped 1xN image AND im_vet_float: im_vet as float.  
def image_preparation(im):
  im_vet = im.reshape((-1, 3))
  im_vet_float = np.float32(im_vet)
  return im_vet, im_vet_float


# converts the image from bgr colorspace into rgb colorspace. Necessary because image format of CV2 is BGR. Image format of np is RGB. 
# input is im_original: a loaded image (2D array); cococo: colorspace conversion code
def change_colorspace(im_original, cococo):
    im = cv2.cvtColor(im_original, cococo)
    return im

  