import skimage
import skimage.io as io
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import clustering.kmeans_detailed as clust

#set filename ________________________________________________

filename2 = 'C:\\Users\\User\\Desktop\\SS20\\DataScience\\Images\\CellNuclei_Segmentation_data\\data\\BBBC020_v1_images\\BBBC020_v1_images\\jw-1h 1\\jw-1h 1_(c1+c5).TIF'
filename = os.path.join("C:\\","Users", "User", "Images", "jw-1h 2_c5.TIF")

#check if filename lead to a valid file path or not. 
print(filename)
if os.path.isfile(filename):
  print("This is a file")
  img_np = io.imread(filename)
  io.imshow(img_np)
  #io.show()

else:
  print("File not valid.")

#crop the image to size 512,512
img_np = img_np[0:512,0:512]
img_np.shape
print(img_np.size)

#make a copy of the picture with only the channel 'blue' values
img_blue = np.copy(img_np[:,:,2])

#set a treshold for removing black values and selecting the core blue of the nuclei. 
treshold = 150
# Every datapoint below treshold will be black. Every datapoint above treshold will be white. 
    # nuclei will appear as white spots on black background. 
img_blue[img_blue < treshold] = 0
img_blue[img_blue >= treshold] = 255

# obain the indices of white values in img_blue. 
  # nuclei x are the x position, nucleiy the y position. One x corresponds to one y position to determine an index of white. 
    # output is a boolean vector of true (255) or false(not 255). 
nucleix, nucleiy = np.where(img_blue == 255)

print(nucleix)
print(nucleiy)
print(nucleix.shape)

#make the final index vector by joining the nucleix x values and nucleiy y values along the columns. 
nuclei = np.concatenate((nucleix, nucleiy), axis = 0)
# reshape into the form of 2 columns (x,y)
nuclei = np.transpose(nuclei.reshape([2, len(nucleix)]))
print(nuclei)

#prepare for kmeans. Feed the true/false vector into the kmeans function.
eps = 0.1
max_iter = 2000

centr = np.array([[0,100], [300,355], [400,255], [511,100], [300, 500]])
centr, closest_cluster = clust.julia_kmeans(nuclei, centr, eps, max_iter, clust.dist_euclidean)

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


#$draw the points after clustering.
plt.subplot(2,2,4)
for i in range(len(centr)):
  cluster_0 = nuclei[closest_cluster == i]
  plt.scatter(cluster_0[:,0], cluster_0[:,1])
plt.scatter(centr[:,0], centr[:,1])
plt.title('Points after applied clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
