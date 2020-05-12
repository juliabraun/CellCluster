# Load two images into python and stack them

#HERE import ____________________________________________________
import imageio
import cv2
import matplotlib.pyplot as plt
import numpy as np

#here the GLOBAL specifications ________________________________________________
#specify filepath, use double slash !!
im1 = imageio.imread('C:\\Users\\User\\Desktop\\SS20\\DataScience\\Images\\CellNuclei_Segmentation_data\\data\\BBBC020_v1_images\\BBBC020_v1_images\\jw-1h 1\\jw-1h 1_c1.TIF')
im2 = imageio.imread('C:\\Users\\User\\Desktop\\SS20\\DataScience\\Images\\CellNuclei_Segmentation_data\\data\\BBBC020_v1_images\\BBBC020_v1_images\\jw-1h 1\\jw-1h 1_c5.TIF')

alpha = 1
beta = 1
gamma = 0
im_overlay = cv2.addWeighted(im1, alpha, im2, beta, gamma)


#how to plot all images in the same window?? As subplots?
plt.imshow(im1)

plt.imshow(im2)

plt.imshow(im_overlay)
plt.show()
