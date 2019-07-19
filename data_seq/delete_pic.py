# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:35:58 2018

@author: chien
"""
import os
# print(os.getcwd())

# the file you want to delete RGB picture
targetpath = r'C:\Users\adm\Desktop\士恩期末影片\sit\cocoimg1'


file = os.listdir(targetpath)
# print(file)
'''
i=0
deletelist = []
for f in file:
#     i+=1

    fullpath = os.path.join(targetpath,f)
#     print(fullpath)
#     print(fullpath[len(fullpath)-5] )
    
    if (fullpath[len(fullpath)-5] == '3' ):
        if(len(deletelist) !=0):
            for d in range(1,len(deletelist)):
                deletefile = deletelist.pop() 
                print('delete the picture : ',deletefile)
                os.remove(deletefile)
        deletelist.clear()
    else:
        deletelist.append(fullpath)
#         print(fullpath)
    
        
#     if (i>10):
#         break
'''


count = 0
os.chdir(targetpath)
allfile =  os.listdir(targetpath)

for f in allfile :
    currname = os.path.join(targetpath, f)

    if currname[-5]== "3" :
        print(currname)
        os.remove(currname)
    else :
        os.rename(currname, "train" + str(count) + ".jpg")
        count += 1
