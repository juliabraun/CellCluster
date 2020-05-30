r"""
This package brings functions for clustering data points, based
on a given set of points and a given set of cluster center points.

content (files):
    - __init__.py: Initialize the package clustering. 

    - distance.py: functions that compute distances 
        dist_manhattan(), dist_euclidean(), dist_colorweight()
    - distance_documentation_v1.py: detailed commentary on distance.py

    - kmeans_detailed.py: functions that make a kmeans clustering
        assign_closest_cluster(), update_centr(), accuracy(), kmeans()
    - kmeans_detailed_documentation_v1.py: detailed commentary on kmeans_detailed.py

    - testing.py: functions that create synthetic defined point clouds for testing the clustering
        

import commands:
    - import clustering.distance as distance
    - import clustering.kmeans_detailed as clust
    - import clustering.testing as testing

"""
