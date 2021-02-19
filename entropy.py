import nibabel as nb
import os
from skimage import data
from skimage.measure.entropy import shannon_entropy
from skimage.color import rgb2gray
import numpy as np

path = "/Users/debodeepkar/Documents/ADNI/NORMALIZED/ANTS/CN/"
os.chdir(path)
file = os.listdir(path) #returns a list with path

slice=[]
A=[]
B=[]

for i in file:
    #i = str(i)
    img = nb.load(i)
    ent=[]
    #name=np.append(name,i)
    for i in range(0,img.shape[2]):
        
        org=img.get_data()[:,:,i]
        gray=rgb2gray(org)
        entr=shannon_entropy(gray)
        ent.append(entr)
    slice=np.append(slice,np.argmax(ent))
    x = np.argmax(ent)
    a = x-5
    b = x+5
    A=np.append(A,a)
    B=np.append(B,b)
    
print(slice) # prints the slice number with highest entropy
print(file) #prints the respective file name
