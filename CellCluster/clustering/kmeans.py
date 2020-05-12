import cv2
import numpy as np
import matplotlib.pyplot as plt


# kmeans clustering. 
 # input is: k: number of clusters; 
 # attempts: number of times cv2.kmeans is executed, each with different initial conditions. 
 # fc: choose a flag to specify how to find the intial center. 0: KMEANS_RANDOM_CENTERS, 1: KMEANS_PP_CENTERS (kmeans ++ center initialisation)
 # tc: choose a termination to specify when the algorithm will stop. 0: TERM_CRITERIA_EPS, 1: TERM_CRITERIA_MAX_ITER, 2: both.
 # max_iter: maximum number of iterations of point assignment in one round of the algorithm
 # epsilon: accuracy
 # im_vet_float: the reshaped, float, 1xN image array (N is the array's lenght)

# output is: 
 # compactness: the sum of squared distances of each point/pixel to its corresponding cluster-center. The smaller the better. 
 # label: an array of numbers that assigns to each pixel (by index) the corresponding cluster-number. 
 # center: list of lists of R,G,B values for each cluster-center. 
def kmeans(k, attempts, fc, tc, max_iter, epsilon, im_vet_float):
  flag_choose = [cv2.KMEANS_RANDOM_CENTERS, cv2.KMEANS_PP_CENTERS]
  flags = flag_choose[fc]
    
  termination_choose = [cv2.TERM_CRITERIA_EPS,  cv2.TERM_CRITERIA_MAX_ITER, cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER]
  termination = termination_choose[tc] 
    
  criteria = (termination, max_iter, epsilon)

  compactness, label, center = cv2.kmeans(im_vet_float, k, None, criteria, attempts, flags)
  return compactness, label, center


# reconstructing the image after kmeans-clustering based segmentation. 
# input is: label: label array output of kmeans()
  # center: center output of kmeans()
  # im: the original image whose shape should be copied
# output is: im_segmented: segmented image with original shape
# steps are:
  # convert the floats of center back to 8-bit pixel values [[R,G,B], [R,G,B]]. Each sublist in the list corresponds to a cluster-center. 
  # flatten the labels array
  # make each pixel (given by index a) the colour of its corresponding cluster-center (as indicated by the label at index a)

def image_reconstruction(label, center, im):
  center = np.uint8(center)
  label_flat = label.flatten()

  im_segmented = center[label_flat]
  im_segmented = im_segmented.reshape(im.shape)

  return im_segmented


# plotting the segmented image and the original in individual subplots within a larger plot. 
# inputs: k: number of clusters
  # figure_size: figure size
  # im: original image after color code conversion
  # im_segmented: image after kmeans() and image_reconstruction()
def image_visualization(k, figure_size, im, im_segmented):
  plt.figure(figsize = (figure_size, figure_size))
 
  #plot original image
  plt.subplot(1,2,1), plt.imshow(im)
  plt.title('Original image'), plt.xticks([]), plt.yticks([])

  #plot for k
  plt.subplot(1,2,2), plt.imshow(im_segmented)
  plt.title('Segmented image when k = %i' %k), plt.xticks([]), plt.yticks([])

  plt.show()