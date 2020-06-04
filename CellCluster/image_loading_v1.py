r""" This startup file loads an image, modifies it 
and applies a kmeans clustering to it. 
Write functions: treshold(), rescaling()
"""
import skimage
import skimage.io as io
import skimage.transform as transform
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import clustering.kmeans_detailed as clust
import clustering.distance as distance
import IO.load_image as loader
import maxcolor
# how to call a function from a package ????????????????????
import preprocessing.radius as radius


r""" Here the filepath to the image used in all next steps stored. 
This section will be replace by filepath 
"""
#set filename ________________________________________________
filename = os.path.join("C:\\","Users", "User", "Images", "jw-1h 2_c5.TIF")

r""" Here some preprocessing is done. The aim of a clustering is 
to identify the relevant information in the picture, which is
the information with highest variation. In the case of cell nuclei,
the highest variation is the color of the nuclei, and the position
of the nuclei. The position of nuclei can be found by finding the
position of the nuclei color. The background is not necessary to 
find the nuclei. 

"""

r""" parameters 
set a treshold for removing black values and selecting the core blue of the nuclei. 
"""
treshold = 100
r = 15
img_np = loader.check_filepath(filename)

r""" Facultative
Here are facultative operations for making the image smaller. 
These can be cropping the image, but also rescaling the image. 
This is done to speed up the execution of this file. 
"""
#crop the image to size 512,512
#img_np = img_np[0:1024,0:1024]


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

img_channel_2 = np.copy(img_np[:,:,2])
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
img_channel_2 = np.uint8(255*transform.rescale(img_channel_2, 0.25, anti_aliasing=False))

r"""
Here, a 3D image is reconstructed. 
This has to be done because the subsequent functions have been written for 
a 3D input. The image should be the same size as img_channel_2, but with 
3 channels instead of 1. These additional channels should be filled with zeros. 
The object type is set to uint8 because subsequent functions need integers. 
"""
img_np = np.uint8(np.zeros([img_channel_2.shape[0], img_channel_2.shape[1], 3]))
r"""
Here, the blue channel (2) of the still empty reconstructed image is filled
with the blue channel values stored in img_channel_2. 
"""
img_np[:,:,2] = img_channel_2



r"""
This step is essential to find out the initial center points for clustering. 
Knowing the average radius of the nucleus, it will be possible to mark one 
cell after the other; thereby finding out the number of cells and the position. 
"""
radius.estimate_radius(img_channel_2)

r""" Treshold and find the pixels corresponding to a nucleus. 
A tresholding is applied to select the region which contains 
relevant information about the nuclei. Background information 
is not relevant, so will be set to 0 (black). Everything above treshold 
is relevant and will be set to 255 (white). 
Nuclei will appear as white spots on black background. 
Note:
the treshold value has to be sufficiently close to the color of 
the nuclei. There will be a variation in intensity values, for ex
values between 255-200. The tresholding uniforms all these values
to obtain a black-and-white-image of only 0 and 255. 
This is to simplify the selection of the cells (as all pixels with 
intensity 255). 
"""
img_channel_2[img_channel_2 < treshold] = 0
img_channel_2[img_channel_2 >= treshold] = 255



# obain the indices of white values in img_channel_2. 
  # nuclei x are the x position, nucleiy the y position. One x corresponds to one y position to determine an index of white. 
    # output is a boolean vector of true (255) or false(not 255). 
nucleix, nucleiy = np.where(img_channel_2 == 255)
nucleic = img_np[nucleix, nucleiy, 2]/255.0
print(nucleic)

print(nucleix)
print(nucleiy)
print(nucleix.shape)

#make the final index vector by joining the nucleix x values and nucleiy y values along the columns. 
nuclei = np.concatenate((nucleix, nucleiy, nucleic), axis = 0)
# reshape into the form of 2 columns (x,y)
nuclei = np.transpose(nuclei.reshape([3, len(nucleix)]))
print(nuclei)

#prepare for kmeans. Feed the true/false vector into the kmeans function.
eps = 0.1
max_iter = 2000

#centr = np.array([[0,100,5], [300,355,20], [400,255,5], [511,100,5], [300, 500,20]])

# create the vector of centers! 
centr = maxcolor.create_centers(img_np, r, treshold)
centr, closest_cluster = clust.kmeans(nuclei, centr, eps, max_iter, distance.dist_colorweight)




#make a new figure:
plt.figure()

#plot original image
plt.subplot(2,2,1)
plt.title("Original image")
io.imshow(img_np)

# plot histogram logarithmic scale
plt.subplot(2,2,2)
histB, bin_edges = np.histogram(img_np[:,:,2], bins = range(256))
plt.title("Histogram of channel B values")
plt.bar(bin_edges[:-1], histB, width = 1)
plt.yscale("log")
plt.plot()

#plot tresholded and channel blue selection
plt.subplot(2,2,3)
io.imshow(img_channel_2)
plt.title('After tresholding and boolean conversion')

theta = 90*np.pi/180
centrplot = loader.rotate_vet(theta,centr)

#$draw the points after clustering.
plt.subplot(2,2,4)
for i in range(len(centr)):
  cluster_0 = nuclei[closest_cluster == i]
  cluster_0_rot = loader.rotate_vet(theta,cluster_0)
  plt.scatter(cluster_0_rot[:,0], cluster_0_rot[:,1])
plt.scatter(centrplot[:,0], centrplot[:,1])
plt.title('Points after applied clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
