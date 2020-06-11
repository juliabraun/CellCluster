# ---- INFO -----
#This is the version with concise comments. For detailed comments,
#see file image_loading_documentation_v1.py.

#This startup file loads an image, modifies it 
#and applies a kmeans clustering to it. 

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
import pathlib as pl
import IO.display as display


# set filename ________________________________________________
filepath = pl.Path("C://","Users", "User", "Images", "jw-1h 3_c5.TIF")
img_np_original = loader.check_filepath(filepath)

# make a copy of the original image. img_np with be the image to work with. 
img_np = np.copy(img_np_original)



# TEST SPACE


   

img_test = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2]))


display.figures_result((img_test, "ImageA", "Points in axis0", "Points in axis1"), (img_test, "ImageB", "axis0", "axis1"))
# END TEST SPACE


# rescale the image and extract one channel
img_np, img_channel = image_manipulation.img_rescaling(img_np, 2, 0.1)

# This step is essential to find the initial center for kmeans clustering. 
# Knowing the average radius of the cells, it will be possible to iterate 
# over one after the other. 
r = radius.estimate_radius(img_channel)
# adjust the radius to be used for the center estimation.
# It should be large enough to not find two centers in the same 
# nucleus, but small enough to not find one center for two nuclei. 
radius_find_center = 2.2*r

# remove all values below treshold; obtain only values corresponding 
# to nuclei. 
nuclei = image_manipulation.treshold_values(img_channel, img_np, 60)

# Initialise kmeans
# The intial centers can be found with the function find_centers.create_centers().
# find_centers is based on finding the maximum intensity. 
# Once the initial center points are found, they are fed to the kmeans(). 
# kmeans() image_manipulation.ist for a distance function, here the weighted distance. 

eps = 0.1
max_iter = 2000
centr = find_centers.create_centers(img_np, radius_find_center, 60)
centr, closest_cluster = clust.kmeans(nuclei, centr, eps, max_iter, distance.dist_colorweight)



# plot tresholded and channel blue selection
plt.subplot(2,2,3)
io.imshow(img_channel)
plt.title('After tresholding')

theta = 90*np.pi/180
centrplot = image_manipulation.rotate_vet(theta,centr)

# draw the points after clustering.
plt.subplot(2,2,4)

for i in range(len(centr)):
  cluster_0 = nuclei[closest_cluster == i]
  cluster_0_rot = image_manipulation.rotate_vet(theta,cluster_0)
  plt.scatter(cluster_0_rot[:,0], cluster_0_rot[:,1])
plt.scatter(centrplot[:,0], centrplot[:,1])
plt.title('Points after applied clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
