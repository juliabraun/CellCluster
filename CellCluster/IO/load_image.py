import skimage
import skimage.io as io
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import clustering.kmeans_detailed as clust
import pathlib as pl


#check if filename is a valid file path or not. 
def check_filepath(file_path: pl.Path):
    if file_path.exists():
        img_np = io.imread(file_path.absolute())
        return img_np
    else:
        raise FileNotFoundError(f"File {file_path} is not a file ")
    return img_np

