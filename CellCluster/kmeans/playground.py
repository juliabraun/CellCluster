#for grayscale: 2 numbers, row and column; for RGB red green blue: also the amount of channels. 
  #print(im.shape)

#select the total number of pixels of the image
  #print(im.size)


#split image into the channels with selecting the channel, then flatten the selected np.array
  #r = im[:,:,0]
  #g = im[:,:,1]
  #b = im[:,:,2]
  #flat_r = r.flatten() 
  #flat_b = b.flatten()
  #flat_g = g.flatten()
  #print(g)
  #print(flat_g)


#load image and display image
  #create a 3D scatter plot of the three channels
  #fig =  plt.figure()
  #ax = Axes3D(fig)
  #ax.scatter(r,g,b)
  #plt.show()


 
  
 
  #plt.imshow(im)
  #plt.show()


r""" This a testing setup for concatenation, reshaping and transpose. 

"""
veta = np.array([0,1,2,3])
vetb = np.array([4,5,6,7])
vetc = np.array([8,9,10,11])

# find out how concatenation works
vet_all = np.concatenate((veta, vetb, vetc), axis = 0)
print(vet_all)

# find out how reshape works
vet_43 = np.reshape(vet_all, (3,4))
print(vet_43)
# check if shape is like it should be 
print(vet_43[0,:])
print(vet_43[:,0])

# find out how transpose works
vet_transposed = np.transpose(vet_43)
print(vet_transposed[0,:])
print(vet_transposed[:,0])





r""" This is how to make a black-and-white-picture. 
It may be useful for dicescore testing. 
Treshold and find the pixels corresponding to a nucleus. 
A tresholding is applied to select the region which contains 
relevant information about the nuclei. Background information 
is not relevant, so will be set to 0 (black). Everything above treshold 
is relevant and will be set to 255 (white). 
Nuclei will appear as white spots on black background. 
Note:
the treshold value has to be sufficiently close to the color of 
the nuclei. There will be a variation in intensity values, for ex
values between 255-200. The tresholding uniforms all these values
to obtain a black-and-white-image of only 0 and 255. 
This is to simplify the selection of the cells (as all pixels with 
intensity 255). 
"""
img_channel_2[img_channel_2 < treshold] = 0
#img_channel_2[img_channel_2 >= treshold] = 255