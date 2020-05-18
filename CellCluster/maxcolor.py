import IO.load_image as loader
import os
import numpy as np
import skimage.io as io
import mask as mk
import matplotlib.pyplot as plt


def create_centers(img_np_colored, r, color_treshold):
#create extended image + 10 pixels at each side
#r = 50
  img_np = np.copy(img_np_colored[:,:,2])
  ext1 = img_np.shape[0]+2*r
  ext2 = img_np.shape[1]+2*r
  img_ext = np.zeros((ext1, ext2))

  left_index = (r,r)
  right_index = (img_ext.shape[0]-r, img_ext.shape[1]-r)


  img_ext[left_index[0]:right_index[0], left_index[1]:right_index[1]] = img_np
  #io.imshow(img_ext)
  #io.show()
  #print(img_ext)


  #find maximum color in img_np
  mask = mk.circular_mask(r)

  maxcol = 255
  save_c_max = []
  while maxcol > color_treshold:
    maxcol = np.amax(img_ext)
    #print(maxcol)
    img_whitex, img_whitey = np.where(img_ext == maxcol)

    first = (img_whitex[0], img_whitey[0])

    left_index = (first[0]-r, first[1]-r)
    right_index = (first[0]+r, first[1]+r)
    submattochange = img_ext[left_index[0]:right_index[0], left_index[1]:right_index[1]]
    img_ext[left_index[0]:right_index[0], left_index[1]:right_index[1]] = np.multiply(submattochange,mask)
    #io.imshow(img_ext)
    #io.show()
    list_save = [first[0]-r, first[1]-r, maxcol]
    save_c_max.append(list_save)

  print(img_np.shape)
  plt.figure()
  save_c_max = np.int32(np.array(save_c_max))
  print(save_c_max.shape)
  i = 0
  while i < save_c_max.shape[0]:
    img_np_colored[save_c_max[i,0], save_c_max[i,1], 0] = 255
    i = i+1

  io.imshow(img_np_colored)
  io.show()
  return save_c_max
