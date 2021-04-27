import nibabel as nb
import os
from skimage import data
from skimage.measure.entropy import shannon_entropy
from skimage.color import rgb2gray
import numpy as np
import shutil


# q= 75/folder-path

def entropy_percent(p,q,r,x):
    path = p # 3d folder path
    os.chdir(path)
    file = os.listdir(path) #returns a list with path 
    src = r # opened slice path
    l=0
    slice=[]
    A=[]
    B=[]
    

    for i in file:
        ent=[]
        #i = str(i)
        img = nb.load(i)

        #name=np.append(name,i)
        for k in range(0,img.shape[2]):

            org=img.get_data()[:,:,k]
            gray=rgb2gray(org)
            entr=shannon_entropy(gray)
            ent.append(entr)
        slice=np.argmax(ent)   #max Entropy slice number
        max = ent[slice]
        y = x * max
        
       # for k in ent:
          #  print(k)
      #  print("end")
            

       
        for j in ent:
            i=str(i)
            j=float(j)
            if j == y or j > y:
                l=ent.index(j)
                if(len(str(abs(l)))==2):
                    #j=str(j)
                    #j=("0"+j)
                    #print(ent.index(j))
                    l=ent.index(j)
                    
                    l=str(l)
                    l=("0"+l)
                    #print(l)
                    name = (src+"\\"+i[:-4]+"_s"+l+".png")
                    shutil.copy(name,q) # create new folder
                else:
                    #print(ent.index(j))
                    l=ent.index(j)
                    l=str(l)
                    #print(l)
                    name = (src+"\\"+i[:-4]+"_s"+l+".png")
                    shutil.copy(name,q)

entropy_percent(r"D:\ADNI\5-fold_CV\3d_files\Folder-1\AD\\", r"D:\ADNI\5-fold_CV\80\saggital\folder-1\AD\\",r"D:\ADNI\5-fold_CV\images-2d\saggital\Folder-1\AD",0.80)