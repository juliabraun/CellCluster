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
import kmeans.clustering.kmeans_detailed as clust
import kmeans.clustering.distance as distance
import kmeans.IO.load_image as loader
import kmeans.image_processing.image_manipulation as image_manipulation
import kmeans.image_processing.image_properties as image_properties
import kmeans.clustering.find_centers as find_centers
import kmeans.image_processing.radius as radius
import pathlib as pl
import kmeans.IO.display as display


# set filename ________________________________________________
#filepath = pl.Path("C://","Users", "User", "Images", "jw-1h 3_c5.TIF")
#img_np_original = loader.check_filepath("C:\\Users\\User\\Images")
img_np_original = io.imread("C:\\Users\\User\\Images\\jw-1h 3_c5.TIF")

# greeting
print("Welcome. This program performs a kmeans clustering on points of a set. \
It computes: \n \
1. the average radius of the nuclei. \n \
2. the number of nuclei in a given image. \n \
3. the accuracy after each round of clustering \n \
It displays: \n \
3. the original image \n \
4. the found centers of each cluster, as red pixel in the original image \n \
5. each recognised nucleus colored in its own color. Different color, different cluster. \n")

# make a copy of the original image. img_np with be the image to work with. 
img_np = np.copy(img_np_original)


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

# Initialise kmeans ______________________________________________
# The initial centers can be found with the function find_centers.create_centers().
# find_centers() is based on finding the maximum intensity. 
# Once the initial center points are found, they are fed to the kmeans(). 
# kmeans() uses image_manipulation.dist as a distance function, here the weighted distance. 

eps = 0.1
max_iter = 2000
centr = find_centers.create_centers(img_np, radius_find_center, 60)

# count the nuclei per image (equal to the number of centers for clustering)
image_properties.count_nuclei(centr)

# apply kmeans to assign points of the set to center points
centr, closest_cluster = clust.kmeans(nuclei, centr, eps, max_iter, distance.dist_colorweight)

# color each cluster differently. Preserve the intensity differences 
# of each nucleus. 
img_colored = display.color_clusters(nuclei, img_np, closest_cluster)

description_1 = (img_np_original, "Original image", "axis0", "axis1")
description_2 = (img_np, "Estimated centers marked red", "", "axis1")
description_3 = (img_colored, "Different color, different cluster", "axis0", "axis1")
display.figures_result(description_1, description_2, description_3)