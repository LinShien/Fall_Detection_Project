# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:51:02 2018

@author: adm
"""
import os

# mypath ='C:\\Users\\adm\\Desktop\\11_03\\11'
# heatdir = r"C:\Users\adm\Desktop\11_03\cocoimg???"#####自己決定
# RGBdir = heatdir+"rgb"
# assert not os.path.isdir(heatdir)##避免重複
# assert not os.path.isdir(RGBdir)
# os.mkdir(RGBdir)
# os.mkdir(heatdir)

# the original file
################################################
#the file mixed the heat img and RGYimg , we rename "cocoimg?"
count = 0
dir_ = r"C:\Users\Lin_Shien\Desktop\data_seq\walk\cocoimg9"
os.chdir(dir_)
allfile = os.listdir(dir_)

for file in allfile:
    currname = os.path.join(dir_, file)
    if currname[-5] == '3' :
        os.remove(currname)
    else :
        os.rename(currname, "train" + str(count) + ".jpg")
        count += 1
#the newfile 
# the new file store RGY img exampe rename that cocoimgRGB?
newdir = r"C:\Users\adm\Desktop\士恩期末影片\fall\cocoimgRGB1"
'''
if not os.path.isdir(newdir):
    os.mkdir(newdir)


for fname in allfile:
    currname = os.path.join(dir_,fname)
    print(currname)
    if fname[-7:-4]=="RGB":
        count = fname[5:-7]
        if str(count)!=5:
            newname = str(count)+".jpg"
            newname = os.path.join(newdir,newname)
        os.rename(os.path.join(newdir,newname) , newdir )

    else:
        count = fname[5:fname.find('.')]
        if str(count)!=5:
            newname = "train" + str(count)+".jpg"
            newname = os.path.join(dir_,newname)
    #print(newname)
    os.rename(currname ,newname  )
'''


