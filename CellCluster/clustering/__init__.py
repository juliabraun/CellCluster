r"""
This package brings functions for clustering data points, based
on a given set of points and a given set of cluster center points.
The documentations to these files are found in the folder "documentation". 

content (files):
    - __init__.py: Initialize the package clustering. 

    - distance.py: functions that compute distances 
        dist_manhattan(), dist_euclidean(), dist_colorweight()

    - find_centers.py: functions that find the initial centers for clustering 
        create_centers(), random_centers()

    - kmeans_detailed.py: functions that make a kmeans clustering
        assign_closest_cluster(), update_centr(), accuracy(), kmeans()

    - testing.py: functions that create synthetic defined point clouds for testing the clustering        

import commands:
    - import clustering.distance as distance
    - import clustering.find_centers as find_centers
    - import clustering.kmeans_detailed as clust
    - import clustering.testing as testing
"""
