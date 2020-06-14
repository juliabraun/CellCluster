
# This file contains functions for displaying. 
# content(files):
#     - color_clusters()
#     - figures_result(*argv)


import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import colorsys as cs
import random


def color_clusters(nuclei, img_np, closest_cluster):
    r"""
    This function replaces intensity values in an image 
    with different colors. It will be used to change the 
    color of each nucleus that has been clustered 
    to a different color. 
    """
    # Debug
    
    # img_np = np.uint8(np.zeros([4, 3, 3]))
      
    # nuclei = np.int32(np.array([[0,1,125], [0,2,125], [2,1,255], [2,2,255]]))
    # ]
    # closest_cluster = np.int32(np.array([0,0,1,1]))
    # centr = np.array([[3,2,25],[2,1,70]])

    # Finish debug
   

    img_colored = np.uint8(np.zeros([img_np.shape[0], img_np.shape[1], img_np.shape[2]]))


    # plt.figure()
    # plt.subplot(2,2,1)
    # attention! axis0 is vertical and axis1 is horizontal. 
    #io.imshow(img_colored)


    # the number of clusters (+1 since amax ignores the zero indexing)
    number_of_clusters = np.amax(closest_cluster) + 1
    # iterate over the number of clusters
    j = 0
    while j < number_of_clusters:
        nucleus_1 = np.where(closest_cluster == j)
        # random color. random() creates a random number 
        # between 0 and 1. Normalisation to 255. 
        # r = random.random()*255
        # g = random.random()*255
        # b = random.random()*255
        # color = (r,g,b)
        
        # create random color for hsv function
        random_c = random.random()

        # iterate over each point belonging to one cluster
        i = 0
        while i < (len(nucleus_1[0])):

            # indexing of the n-th nucleus
            # index of nucleus_n in closest_cluster
            # WARNING! A tuple object is accessed with double brackets [][]
            #index_point_0 = nucleus_1[0][i]
            index_point_0 = nucleus_1[0][i]
            # print(index_point_0)
          
            # index of first point belonging to nucleus
            first_point = np.int32(nuclei[index_point_0,:])

            # index intensity value
            nuclei_intensity = nuclei[index_point_0, 2]
            # print(intensity)
            
            # specify a random color. The intensity variation should be 
            # preserved. hsv is a different way to express colors in a computer
            # (hue, saturation, value)
            color = 255*np.abs(cs.hsv_to_rgb(random_c, 1, np.float(nuclei_intensity)))

            # access element in img_colored, 2 specifies the blue channel
            # replace intensity value of img_colored
            img_colored[first_point[0], first_point[1], :] = color

            # show how one pixel after the other is being colored
            #io.imshow(img_colored)
            #plt.show()
            i = i+1
        # increase counter to next number_of_clusters
        j = j+1

    # plot
    # plt.subplot(2,2,1)
    # io.imshow(img_np)
    # plt.subplot(2,2,2)
    # io.imshow(img_colored)
    # plt.show()

    return img_colored


#img_np = np.uint8(np.zeros([4, 3, 3]))
#nuclei = np.int32(np.array([[0,1,60], [0,2,90], [2,1,230], [2,2,255]]))
#closest_cluster = np.int32(np.array([0,0,1,1]))
#color_clusters(nuclei, img_np, closest_cluster)

def figures_result(*argv):
    r""" This function creates specified input images. 
    - input: *argv: variable number of arguments arg. 
    Arguments must be given as lists with parameters
    belonging to the image, e.g. (img_test, "ImageA"). 
    - Each argument arg must contain: 
    - img_input: the image to be plotted
    - title: str containing the title of the image
    - axis0: str containing the axis0 description
    - axis1: str containing the axis1 description
    
    WARNING:
    The sequence must be kept. 
    argv[i-1][1] must be string if a title should be given. 
    """
    # make a new figure:
    plt.figure()

    # specify the dimensions of the plot 
    # np.sqrt finds the next-closest square length.  
    # np.floor chooses the next-smallest integer if a float results.  
    # dim1 specifies number of columns 
    dim1 = np.int32(np.ceil(np.sqrt(len(argv))))
    # dim2 specifies number of rows. 
    dim2 = dim1

    # adapt the squared figure so that all images will fit. 
    # the number of images should correspond to the product of 
    # the plot dimensions. Increase dim1 by 1 until the condition 
    # is true. 
    while len(argv) >= dim1 * dim2:
        dim1 = dim1 + 1
    
    # first image position in the plot
    i = 1
    # iterate over the arguments to make all plots. 
    for arg in argv:
        # the subplot position is fixed by the integer i
        plt.subplot(dim1,dim2,i)
        # second argument is the title. i-1 is needed since there
        # is zero-indexing but non-zero counting of plots. 
        # argv is indexed as a list. Selects the i-th tuple, and 
        # from that, the second value as title. 
        plt.title(argv[i-1][1])
        plt.xlabel(argv[i-1][2])
        plt.ylabel(argv[i-1][3])
        # choose the image at index 0 of each tuple to be plotted. 
        io.imshow(arg[0])
        # go to next image position in the plot
        i = i+1
    # show the plot with all images
    plt.show()


def make_hist(img_input, channel):
    r""" This function plots a histogram. 
    - input:
        - img_input: image to be histogrammed
        - channel: channel to be chosen for histogram
    """
    #plt.subplot(2,2,2)
    # histB contains the count of how many times an intensity
    # value occurs. bin_edges contains the start and end point of each bin. 
    histB, bin_edges = np.histogram(img_input[:,:,channel], bins = range(256))
    plt.title("Histogram of channel %s values" %(str(channel)))
    plt.bar(bin_edges[:-1], histB, width = 1)
    plt.yscale("log")
    plt.plot()
    plt.show()


def plot_after_clustering(centr, nuclei, closest_cluster):
    r"""This function draws the points after clustering.
    """
    plt.subplot(2,2,4)
    theta = 90*np.pi/180
    centrplot = image_manipulation.rotate_vet(theta,centr)
    # for all centers. len(centr) corresponds to the number of nuclei. 
    for i in range(len(centr)):
        # find all points corresponding to one nucleus = one cluster
        # and store in cluster_0 as axis0, axis1, intensity value. 
        cluster_0 = nuclei[closest_cluster == i]
        #print(cluster_0)

        # rotate the set of points, to be displayed similar to 
        # the original image 
        cluster_0_rot = image_manipulation.rotate_vet(theta,cluster_0)
        # plot all points as axis0 against axis1 values
        plt.scatter(cluster_0_rot[:,0], cluster_0_rot[:,1])
    # each point is a list of three values. [:,0] select all points, 
    # and the first axis0 value of each. 
    # [:,1] select all points, and the axis1 value of each. 
    plt.scatter(centrplot[:,0], centrplot[:,1])
    plt.title('Points after applied clustering')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return

