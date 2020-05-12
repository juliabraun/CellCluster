#NEXT STEPS: for loop for kmeans, apply on multiple pictures and show at the same time: list of filepaths (os function): extention _c5.tif




# specify that this module be executed
if __name__ == '__main__':
# Load an image into python and display it

#HERE import ____________________________________________________
  import cv2
  import imageio
  import matplotlib.pyplot as plt
  import numpy as np
  from mpl_toolkits.mplot3d import Axes3D
  from imagemanipulation import load_singleimage_cv2

#here the GLOBAL specifications ________________________________________________
#specify filepath, use double slash !!
  filename = 'C:\\Users\\User\\Desktop\\SS20\\DataScience\\Images\\CellNuclei_Segmentation_data\\data\\BBBC020_v1_images\\BBBC020_v1_images\\jw-1h 1\\jw-1h 1_c5.TIF'
  #scope issue: to access return value from a function, store it in a new variable. Return values inside functions are not available to global scope. 
  im_original = load_singleimage_cv2(filename)

  #im_original = cv2.imread(filename)
  #im2 = imageio.imread('C:\\Users\\User\\Desktop\\SS20\\DataScience\\Images\\test.png')
  im = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)

  
  #change the shape from an image into a MxN matrix??? reshape(channels, rows)
  vectorized = im.reshape((-1, 3))
 
  #convert from unit8 to float: cv2.kmeans() need float as input
  vectorized = np.float32(vectorized)
  print(vectorized)


  #KMEANS Clustering________________________________
  #define criteria for iteration termination, criteria = (type: maxiter, eps, or both; maxiter: integer, epsilon: accuracy)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

  #attemps are number of repetitions of the algorithm with different starting points
  #flags specify how starting centroids are chosen: random centers, or center initialisation
  k = 2
  attempts = 10
  flags = cv2.KMEANS_RANDOM_CENTERS

  #KMeans output: compactness (distance of each pixel to centroid), labels (list to which cluster each pixel belongs), centers (all centers of clusters)
  compactness, label, center = cv2.kmeans(vectorized, k, None, criteria, attempts, flags)

  #KMEANS OUTPUT manipulation____________________________________________
  #convert to uint8, WHY is the CENTER so important??
  center = np.uint8(center)

  #access labels as 1D array (flattened)
  res = center[label.flatten()]

  #reshape image into the original image form 
  result_image = res.reshape(im.shape)

  #visualize the output__________________________________________________________
  figure_size = 15
  plt.figure(figsize = (figure_size, figure_size))
  
  #plot original image
  plt.subplot(1,2,1), plt.imshow(im)
  plt.title('Original image'), plt.xticks([]), plt.yticks([])

  #plot for k = k[0]
  plt.subplot(1,2,2), plt.imshow(result_image)
  plt.title('Segmented image when k = %i' %k), plt.xticks([]), plt.yticks([])

  plt.show()


#for grayscale: 2 numbers, row and column; for RGB red green blue: also the amount of channels. 
  #print(im.shape)

#select the total number of pixels of the image
  #print(im.size)


#split image into the channels with selecting the channel, then flatten the selected np.array
  #r = im[:,:,0]
  #g = im[:,:,1]
  #b = im[:,:,2]
  #flat_r = r.flatten()
  #flat_b = b.flatten()
  #flat_g = g.flatten()
  #print(g)
  #print(flat_g)


#load image and display image
  #create a 3D scatter plot of the three channels
  #fig =  plt.figure()
  #ax = Axes3D(fig)
  #ax.scatter(r,g,b)
  #plt.show()


 
  
 
  #plt.imshow(im)
  #plt.show()





