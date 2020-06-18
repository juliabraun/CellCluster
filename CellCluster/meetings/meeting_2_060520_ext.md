# Meeting Notes
##### DATE 06/05/2020

## Last weeks progress
- implementing kMeans clustering for image segmentation
   - cv2
   - image loading, preprocessing
   - kmeans
   - image reconstruction, visualization
- research on 'Dice score'
- DataCamp courses on image processing

## Questions
- Is this kmeans coding sufficiently detailed?
- To what extent are we allowed to use functions that are already implemented in certain packages?

## Plans for next week
- devise strategy how to compare segmented image with reference image
- implement dice score
- prepare presentation in english for milestones to get feedback from tutor 
- unit testing: write test functions. function 1 makes dice score; function 2 checks if function 1 is working properly
    - synthetic images: write two identical images, test with dice score --> check if correct 
    - mocking: two different images, test with dice score 
    - small and large differences between images 

## Learned
 - orient on structure of online repository (link available)
 - for the exam: repository will be cloned and executed, environment and images have to be there.  
 - share conda environment: export jaml file datei, put in repository top level, update always when new package installed
 - filepath> import pathlib as pl   pl.path(data//img1.tif), put image data into the repository 
 - use sci-kit image for kmeans algorithm
 - implement kmeans algorithm = write algorithm yourself; or 
 - gitignore file: dont push configuration files, only content files 
 - final report: has to be written as a jupyter notebook. write code in python files .py , and call the files from the notebook (write notebook at end).

