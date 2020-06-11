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
import image_processing.image_manipulation as image_manipulation
import clustering.find_centers as find_centers
import image_processing.radius as radius


r""" Here the filepath to the image used in all next steps stored. 
This section will be replace by filepath 
"""
#set filename ________________________________________________
filename = os.path.join("C:\\","Users", "User", "Images", "jw-1h 2_c5.TIF")
img_np_original = loader.check_filepath(filename)

# make a copy of the original image. img_np with be the image to work with. 
img_np = np.uint8(np.zeros([img_np_original.shape[0], img_np_original.shape[1], img_np_original.shape[2]]))
img_np[:,:,:] = img_np_original[:,:,:]

r""" Here some image_processing is done. The aim of a clustering is 
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
np.uint8 is necessary because np.zeros produces a float by default, but for 
further computation integer (intensity values are needed). 
"""
img_np = np.uint8(np.zeros([img_channel_2.shape[0], img_channel_2.shape[1], 3]))

r"""
Here, the blue channel (2) of the still zeroed reconstructed image is filled
with the blue channel values stored in img_channel_2. 
"""
img_np[:,:,2] = img_channel_2


r"""
This step is essential to find out the initial center points for clustering. 
Knowing the average radius of the nucleus, it will be possible to mark one 
cell after the other; thereby finding out the number of cells and the position. 
The algorithm is executed here before tresholding to improve the accuracy of the 
result. 
"""
radius.estimate_radius(img_channel_2)



r""" prepare set of points pos
The input pos of kmeans() should contain the points of the image, 
but only the most relevant information to identify a nucleus.
The image has been channel_extracted, the average radius of the cell
estimated, the channel_extracted image tresholded to an image of two 
different intensity values: 0 and 255. 
Relevant for the kmeans clustering are the points corresponding to 
each nucleus. The background is irrelevant for the clustering. 

Fed to kmeans() will be pos, a np.array containing the indices that 
refer to each point of a nucleus. Not all indices of the image will be 
inside the resulting pos, the background indices will be neglected. 
Thus, the task is, obtain the indices of white values (255) in img_channel_2. 

nuclei_axis0 contains the axis0 indices, nuclei_axis1 the axis1 
indices. This is because np.where gives the indices separately for each dimension. 
One axis0 index and the corresponding axis1 index (in the same position 
in nuclei_axis0 and nuclei_axis1) determine an index of a white pixel (value 255). 

We need a vector that contains the intensity values of the reconstructed image img_np 
(the image that contains 3 channels, of which only the blue channel has values non-zero.
It contains all the points that belong to a nucleus.)
"""
nuclei_axis0, nuclei_axis1 = np.where(img_channel_2 >= treshold)
r""" nuclei_axis2 contains the intensity values in the channel of img_np 
at the specified index. 
The values are divided by 255.0 to normalise the value between 0 and 1, 
and make them float and easy to handle.

"""
nuclei_c = img_np[nuclei_axis0, nuclei_axis1, 2]/255.0

r"""
The final set of points pos is created by joining pos_axis0, 
pos_axis1 and pos_axis2 one after the other, into one sequence of numbers.  
"""
nuclei = np.concatenate((nuclei_axis0, nuclei_axis1, nuclei_c), axis = 0)
print(nuclei)
r"""
The final pos has to have the shape required for the kmeans algorithm.
pos.shape[0] should be the number of points, pos.shape[1] the number of components. 
"""

nuclei_reshaped = nuclei.reshape([3, len(nuclei_axis0)])

print(nuclei_reshaped.shape)
nuclei_transposed = np.transpose(nuclei_reshaped)
#print(nuclei_reshaped.shape)
#print(nuclei.shape)
#print(nuclei_transposed.shape)


r"""
Initialise kmeans
The intial centers can be found with the function find_centers.create_centers().
create_centers() is based on finding the maximum intensity. 
Once the initial center points are found, they are fed to the kmeans(). 
kmeans() uses the argument dist for a distance function, here the weighted distance. 
"""
eps = 0.1
max_iter = 2000
r""" This line is facultative: centr can be fed also as random_centers()
create_centers() uses the average nucleus radius r and a user-specified treshold to 
find out the middle point of each nucleus. 
r is calculated with radius(). 
"""
centr = find_centers.create_centers(img_np, r, treshold)
print()
centr, closest_cluster = clust.kmeans(nuclei, centr, eps, max_iter, distance.dist_colorweight)



r""" Here the plots are made. 
"""
plt.figure()
#plot original image
plt.subplot(2,2,1)
plt.title("Original image")
io.imshow(img_np)

# plot histogram of the intensity values of channel B with logarithmic scale
plt.subplot(2,2,2)
r""" bins indicates the number of bins, here 255 from 1 to 255. 
np.histogram returns histB: an array containing the histogram values. 
bin_edges: the maximum bin number (length(histB) + 1)
"""
histB, bin_edges = np.histogram(img_np[:,:,2], bins = range(256))
plt.title("Histogram of channel B values")
r""" create a barplot, all bins, the height corresponding to the histogram value
at that bin. 
"""
plt.bar(bin_edges[:-1], histB, width = 1)
plt.yscale("log")
plt.plot()

#plot tresholded and channel blue selection
plt.subplot(2,2,3)
io.imshow(img_channel_2)
plt.title('After tresholding')

# the image was rotated. rotate_vet() rotates it back to original
theta = 90*np.pi/180
centrplot = image_manipulation.rotate_vet(theta,centr)

#$draw the points after clustering.
plt.subplot(2,2,4)
r"""
The points were rotated with respect to the original image. 
This for loop iterates over all points and applies the 
function rotate_vet to it. This rotates the points back to normal. 
The points are plotted ... 
"""
for i in range(len(centr)):
    cluster_0 = nuclei[closest_cluster == i]
    cluster_0_rot = loader.rotate_vet(theta,cluster_0)
    plt.scatter(cluster_0_rot[:,0], cluster_0_rot[:,1])

r"""
This adds the center points into the plot. 
"""
plt.scatter(centrplot[:,0], centrplot[:,1])
plt.title('Points after applied clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
