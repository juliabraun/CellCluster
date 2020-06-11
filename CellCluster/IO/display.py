
# This file contains functions for displaying. 
# content(files):
#     - color_clusters()
#     - figures_result(*argv)


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

    #img_colored[5,70,20]
    #img_colored[selected_cell[iterator,0], selected_cell[iterator,1], :] = color


    # plot
    plt.figure()
    #io.imshow(img_colored)
    #plt.show()
    return

color_clusters()


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
    dim1 = np.int32(np.floor(np.sqrt(len(argv))))
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

