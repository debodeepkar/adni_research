import nibabel as nb
import os
import shutil

path = "/Users/debodeepkar/Documents/ADNI/NORMALIZED/ANTS/CN/" #3d data path
file = os.listdir(path) #returns a list with path

src = "/Users/debodeepkar/Documents/ADNI/NORMALIZED/ANTS/image/CN/" # opened slice path
os.chdir(path)
for i in file:
    img = nb.load(i)
    t = img.get_data().shape[2]
    u = (int((t/2)) - 2)
    l = u+3
    for j in range(u,l):
        i=str(i)
        if(len(str(abs(j)))==2):
            j=str(j)
            j=("0"+j)
            #print(j)
            name = (src+i[:-4]+"_s"+j+".png")
            shutil.copy(name,"/Users/debodeepkar/Documents/ADNI/NORMALIZED/ANTS/sli/CN/") # create new folder
        else:
            j=str(j)
            name = (src+i[:-4]+"_s"+j+".png")
            shutil.copy(name,"/Users/debodeepkar/Documents/ADNI/NORMALIZED/ANTS/sli/CN/")

