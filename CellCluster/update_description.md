date 30.05.2020
# Changes
- replace comments with docstring notation in the files for execution and
change indentation from 2 to 4 spaces
    -- distance.py, kmeans_detailed.py, testing.py

- add description of package clustering in __init__.py (clustering)
- add comments for clustering.testing.py
- rename function: julia_kmeans() to kmeans()
- on github set package structure independent of folder Julia
- test_kmeans.py: execute testing.py functions 

# Next steps
- name save_c_max center_intensity ?
- name submattochange img_squaresubset
- name img_whitex max_intens_x, and img_whitey max_intens_y
- call find_centers find_initial_centers
- change name maxcol to max_intensity
- put mask.py into the appropiate folder. indicate import as mk
- in create_centers() write extend_image()
- fix create_centers() !!!
- write extract_channel()
- write commentary for image_loading_v1.py
- write commentary for io.load_image.py
- write commentary for find_centers_documentation_v1.py
- write commentary for preprocessing.radius.py 
- put preprocessing content of image_loading_v1.py
into file of package preprocessing
- write code for random choice of initial centers: clustering.find_centers.random_centers()
- understand dist_colorweight(ele1, ele2):
- what is def rotate_vet() in load_image?
- modify load_image()
- write function that plots the required details of kmeans


# Introduction to program
To make the clustering program working, 
- the startup file is: image_loading_v1.py
In the startup file at line 24 (ALWAYS UPDATE) you have to replace your own filepath 
leading to the image "jw-1h 2_c5.TIF" which I used for testing. 
In the original folder structure of the dataset, it is stored at "BBBC020_v1_images\jw-1h 2\jw-1h 2_c5.TIF".

To make the testing program working, 
- the startup file is: test_kmeans.py
In lines 13-15 you can choose your own random sets by adjusting the parameters. 

Documentation versions: files with ending 'documentation_v1.py'
The program is set to run with the concise versions, however, 
the code is equivalent in both concise and documentation version. 