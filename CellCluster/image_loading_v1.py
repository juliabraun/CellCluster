
import skimage
import skimage.io as io
import skimage.transform as transform
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import clustering.kmeans_detailed as clust
import clustering.distance as distance
import colordist as coldist
import IO.load_image as loader
import maxcolor


#set filename ________________________________________________

filename2 = 'C:\\Users\\User\\Desktop\\SS20\\DataScience\\Images\\CellNuclei_Segmentation_data\\data\\BBBC020_v1_images\\BBBC020_v1_images\\jw-1h 1\\jw-1h 1_(c1+c5).TIF'
filename = os.path.join("C:\\","Users", "User", "Images", "jw-1h 2_c5.TIF")
#set a treshold for removing black values and selecting the core blue of the nuclei. 
treshold = 100
r = 15
img_np = loader.load_image(filename)

#crop the image to size 512,512
#img_np = img_np[0:1024,0:1024]

img_np.shape
print(img_np.size)

#make a copy of the picture with only the channel 'blue' values
img_blue = np.copy(img_np[:,:,2])
img_blue = np.uint8(255*transform.rescale(img_blue, 0.25, anti_aliasing=False))
img_np = np.uint8(np.zeros([img_blue.shape[0], img_blue.shape[1], 3]))
img_np[:,:,2] = img_blue


# Every datapoint below treshold will be black. Every datapoint above treshold will be white. 
    # nuclei will appear as white spots on black background. 
img_blue[img_blue < treshold] = 0
img_blue[img_blue >= treshold] = 255

# obain the indices of white values in img_blue. 
  # nuclei x are the x position, nucleiy the y position. One x corresponds to one y position to determine an index of white. 
    # output is a boolean vector of true (255) or false(not 255). 
nucleix, nucleiy = np.where(img_blue == 255)
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

centr = maxcolor.create_centers(img_np, r, treshold)
centr, closest_cluster = clust.julia_kmeans(nuclei, centr, eps, max_iter, distance.colordist)




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
io.imshow(img_blue)
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
