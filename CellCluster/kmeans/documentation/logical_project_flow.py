r"""
This startup file gives an overview of how the 
program is structured, and step-by-step what 
it does to make from a picture a kmeans clustering.
The file is divided into different "problems", 
each addressed with a specific part of the clustering. 

Let's start with the input of the kmeans function.
kmeans.clustering.kmeans_detailed.kmeans() has as input, 
because of the mathematical theory behind the following:
pos, centr, eps, max_iter, dist

- pos: numpy.array as matrix containing the set of points.
- centr: numpy.array as matrix containing the set 
  of centers of the clusters. 
- eps: a float representing the accurcacy
- max_iter: an int representing the maximum number of iterations
- dist: a function that returns a positive real number given two arrays. 
  (corresponds here to the distance between point and center point)

Which of these required input are given? 
pos we obtain from the image directly. We can apply certain processing
to the image (like rescaling it, or making it grayscale). In the end,
pos should be a np.array, where 
        pos.shape[0] must be the amount of points. 
        pos.shape[1] is the amount of components defining each point.

What about centr? We need a set of initial clusters chosen in a certain 
way, to start the clustering. The strategy chosen here is, 
      - every nucleus (so, all the points of pos belonging to that nucleus)
      should be clustered in a separate cluster. 
      - to achieve this, each center point should optimally be located in 
      the center of a nucleus.
      - Identifying the centers of the nucleus also determines the 
      position of the nucleus. 

Because of these considerations, the following algorithm has been implemented. 
      - start with a preprocessed image that is only black-and-white: 
        the nucleus pixels are of intensity 255 and 
        the background pixels are of intensity 0. 
      - locate the center of each nucleus by finding the point of highest intensity 
      (functions in the file find_centers.py)
      - replace all the points of pos corresponding to a nucleus with a value of 
      lower intensity. 
      - continue with the next highest intensity point (wich corresponds to the 
      next nucleus)


Problem 1


- load image
- preprocess image (add functions)
- find out initial center points (clustering.find_centers.create_centers())
    - estimate radius
    - 
"""
