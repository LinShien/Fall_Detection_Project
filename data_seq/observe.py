import json
import matplotlib.pyplot as plt
import cv2
import os

with open( r"C:\Users\adm\Desktop\11_03\cocoimg6.json",'r') as f:
    data = json.load(f)

#filepath = r"C:\Users\adm\Desktop\11_03\cocoimg1rgb" #this is rgb
##################the file you want to observe 
filepath = r"C:\Users\adm\Desktop\11_03\cocoimg6" #this is heatimg
##################don't forget to change the filename


info = data['info']#dict
licenses = data['licenses']#dict
images = data['images']#list
annotations = data['annotations']#list
categories = data['categories']#list

img_id=[]
for ann in annotations:
    img_id.append(ann['image_id'])
img_id.sort()


t_id=[]
t_cat=[]
test_dict={}
count = 0



for ann in annotations:
    if (ann['image_id'] == 1):
        test_dict[str(count)] = ann
        del test_dict[str(count)]['segmentation']
        count+=1

img_id.sort()

#heat
#name = 'train' +images[0]['file_name']
#rgb

for t in range(len(annotations)):
    filename = 'train'+str(annotations[t]['image_id'])+'.jpg'
    
    name = cv2.imread(os.path.join(filepath,filename))
    print(filename)
    #for i in range(18):
    for i in range(17):  
        #print(annotations[t]['keypoints'])
        a = annotations[t]['keypoints'][i*3]
        b = annotations[t]['keypoints'][i*3+1]
        cv2.circle(name, (a,b), 15, (255,255,255), thickness=3, lineType=8, shift=0)
    
    plt.imshow(name)
    plt.show()


#######畫點 1 unvisible 2visible

#test = cv2.imread('/Users/chien/Desktop/000000040083.jpg')
#cv2.circle(test, (82,259), 3, (255,0,0), thickness=3, lineType=8, shift=0)
#plt.imshow(test)
#plt.show()


'''
myjson=
{
    "info": info,
    "licenses": [license],
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}


info = {'description': 'COCO Dataset in Thermal Imaging',
        'url': '',
        'version': '1.0',
        'year': 2018,
        'contributor': 'National Sun Yat-sen University ',
        'date_created': '2018/11/06'}

licenses ＝[{'url': 'none',
  'id': 1,
  'name': 'License'},
]


myjson['images']=
{'license': 1,
 'file_name': '000000397133.jpg',
 'coco_url': '',
 'height': 0,
 'width': 0,
 'date_captured': 'none',
 'flickr_url': 'none',
 'id': 0}


myjson['annotations']=[]
ann = 
{'segmentation': [],
 'num_keypoints': 10,
 'area': 47803.27955,
 'iscrowd': 0,
 'keypoints': [0,0,0],
 'image_id': 425226,
 'bbox': [],
 'category_id': 1,
 'id': 183126}




myjson['categories']=[{'supercategory': 'person',
  'id': 1,
  'name': 'person',
  'keypoints': ['nose',
   'left_eye',
   'right_eye',
   'left_ear',
   'right_ear',
   'left_shoulder',
   'right_shoulder',
   'left_elbow',
   'right_elbow',
   'left_wrist',
   'right_wrist',
   'left_hip',
   'right_hip',
   'left_knee',
   'right_knee',
   'left_ankle',
   'right_ankle',
   'Neck'
   ],
  
  'skeleton': [],



'''
