# ideas to writing a KMeans function
# steps needed:
#   1 search k random points (with random intensity value) -> define them as centers, number them
#   2 compute distance between pixel & centers -> label each datdapoint (=pixel) according to the nearest center
#   3 compute the mean (intensity) value in each cluster -> define these mean-values as new centers
#   4 same as 2 -> label pixel according to nearest new center
#   5 check, whether no datapoint changes its cluster or MAX_ITER is reached or a certain accuracy is reached
#       5.1 if all are FALSE -> return to 3
#       5.2 if one of them is true -> go to 6
#   6 return array in shape of image, where each pixel is labelled according to their assigned cluster (numbers)

#   3 to 5 could be done in a while-loop -> tests for MAX-ITER and changed datapoint-labels and accuracy after each
#   iteration

# ideas for code:
#   1: nd.random.randint(0,256), do this k times, store in variables number them from 1 to k OR store in list
#   2: distance(pixel, center) is just difference in intensity value, could be done with a for-loop (for p in image:)
#   2: nearest cluster identified by min() of distances between a pixel and each center
#   2: pixel labelled according to nearest cluster (not in original image, but in array of same shape)
#   3: one could store the indices of all pixels of a cluster in a list, then compute the mean
#   3: define mean as new cluster
#   5: if no datapoint changes cluster: new centers == old centers
#   5: MAX-ITER must be bigger than i, i is increased by 1 after each iteration
#   5: accuracy defined as X, when distance between oldcluster & newcluster) < X, the desired accuracy is reached
#   6: returns array mentioned in 2
