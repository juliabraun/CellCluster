r"""
This file contains functions for displaying. 
"""

import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import colorsys as cs


def color_clusters():
    r"""
    This function replaces intensity values in an image 
    with different colors. It will be used to change the 
    color of each nucleus that has been clustered 
    to a different color. 
    """
    r""" Debug
    """
    img_np = np.uint8(np.zeros([128, 100, 3]))

    nuclei = np.int32(np.array([[100,80,250], [5,6,255], [6,5,255], [6,6,255]]))

    closest_cluster = np.int32(np.array([0,0,1,1]))
    centr = np.array([[70,5,25],[20,5,70]])

    r""" Finish debug
    """

    img_colored = np.uint8(np.zeros([img_np.shape[0], img_np.shape[1], img_np.shape[2]]))



    #for i in range(nuclei.shape[0]):
    #    axis_2_v = nuclei[i,2]
    #    color = cs.hsv_to_rgb(0.5,1,axis_2_v)
    #    img_colored[nuclei[i,0], nuclei[i,1], :] = color

    iterator = 0
    index = np.where(closest_cluster == iterator)
    #print(index)
    selected_cell = nuclei[index,:]
    #print(selected_cell)
    color = cs.hsv_to_rgb(0.5,1,255)
    print(selected_cell[iterator,0])

    img_colored[5,70,20]
    img_colored[selected_cell[iterator,0], selected_cell[iterator,1], :] = color


    # plot
    plt.figure()
    io.imshow(img_colored)
    plt.show()
    return

color_clusters()
