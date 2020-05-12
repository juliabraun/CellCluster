

# Load an image into python and display it

#HERE import ____________________________________________________
import imageio
import matplotlib.pyplot as plt

#here the GLOBAL specifications ________________________________________________
#specify filepath, use double slash !!
im = imageio.imread('C:\\Users\\User\\Desktop\\SS20\\DataScience\\Images\\CellNuclei_Segmentation_data\\data\\BBBC020_v1_images\\BBBC020_v1_images\\jw-1h 1\\jw-1h 1_(c1+c5).TIF')


#load image and display image
plt.imshow(im, cmap = 'gray')



#plot image as histogram NOT WORKING 
plt.hist(im)
plt.show()




from PIL import Image
import glob
image_list = []
for filename in glob.glob('yourpath/*.gif'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)