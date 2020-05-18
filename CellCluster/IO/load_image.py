import skimage
import skimage.io as io
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import clustering.kmeans_detailed as clust
import colordist as coldist


def rotate_vet(theta,vet_i):
  vet_o = np.copy(vet_i)
  vet_o[:,0] = np.cos(theta) * vet_i[:,0] + np.sin(theta) * vet_i[:,1] 
  vet_o[:,1] = - np.sin(theta) * vet_i[:,0] + np.cos(theta) * vet_i[:,1] 
  return vet_o



#check if filename lead to a valid file path or not. 
def load_image(filename):
  if os.path.isfile(filename):
    print("This is a file")
    img_np = io.imread(filename)
    #io.imshow(img_np)
    #io.show()

  else:
    print("File not valid.")
  return img_np

