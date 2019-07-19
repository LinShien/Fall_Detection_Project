#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:30:57 2018

@author: chien
"""

#python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg

import argparse
import logging

#####close logging
logging.disable(logging.CRITICAL)

import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import matplotlib.pyplot as plt

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    '''
        - ArgumentParser -- The main entry point for command-line parsing. As the
        example above shows, the add_argument() method is used to populate
        the parser with actions for optional and positional arguments. Then
        the parse_args() method is invoked to convert the args at the
        command-line into an object with attributes.
    '''

    parser.add_argument('--image', type=str, default='./images/apink3.jpg')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='432*368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    # note :actually the defaulte is 432x368
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    # note : resize-out-ratio is bigger  then the operation time is longer

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    # note: get int from string and need both width and height is mutiple of 16
    # 
    # get_graph_path() is used to get path
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    # 
    

    import os



    # In[] ############################################################################################################################
    ## strat to loop 
    ##start from your picture name , if start from 0 then idcount = 0
    idcount= 0
    annotations=[]
    images=[]
    testtotal=0
    filepath = r"C:\Users\adm\Desktop\11_03\cocoimg1rgb"
    allfile = os.listdir(filepath)
    
    
    logging___ = ""
    for trainname in allfile:
        #print("process the file : " , trainname)
        source_img = os.path.join(filepath, trainname)
        
        #change args.image to s_i
        original_image = cv2.imread(source_img, cv2.IMREAD_COLOR)
        height = original_image.shape[0]
        width = original_image.shape[1]
        image = common.read_imgfile(source_img, None, None)
    

        if image is None:
            logger.error('Image can not be read, path=%s' % args.image)
            sys.exit(-1)
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t

        logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))



        (keypoints , image) = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)


      # In[]    #############################################################################
        info = {'description': 'COCO Dataset in Thermal Imaging',
                'url': '',
                'version': '1.0',
                'year': 2018,
                'contributor': 'National Sun Yat-sen University ',
                'date_created': '2018/11/07'}

        licenses ={'url': 'none','id': 1,'name': 'License'},
    
        categories = {'supercategory': 'person',
                    'id': 1,
                    'name': 'person',
                    'keypoints': 
                    ['nose','left_eye','right_eye','left_ear',
                    'right_ear','left_shoulder','right_shoulder',
                    'left_elbow','right_elbow','left_wrist',
                    'right_wrist','left_hip','right_hip','left_knee',
                    'right_knee','left_ankle','right_ankle','Neck'],
  
                    'skeleton': [[16, 14],[14, 12],[17, 15],[15, 13],[12, 13],
                   [6, 12] ,[7, 13],[6, 7],[6, 8],[7, 9],[8, 10],
                   [9, 11],[2, 3],[1, 2],[1, 3],[2, 4],[3, 5],[4, 6],
                   [5, 7]]}

        image_id = int(trainname[:-4])
        image_ = {'license': 1,
                 'file_name': trainname,
                 'coco_url': '',
                 'height': height,
                 'width': width,
                 'date_captured': 'none',
                 'flickr_url': 'none',
                 'id': image_id}
        images.append(image_)
        
        ##the index of keypoints list
        insert_ = {0:0 , 1:51, 2:18 , 3:24 , 4:30 , 5:15 , 6:21 , 7:27 , 8:36,
                     9:42 ,10:48 ,11:33 ,12:39 ,13:45 ,14:6 ,15:3 ,16:12 ,17:9}

        testtotal+=len(keypoints)
        ##if you want to show the img when label the img

        if(len(keypoints)>1):
            print("#####detect more one:",source_img)
            print("the number of people : ", len(humans))
            #bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            #plt.imshow(bgimg)
            #plt.show()
            '''
        else:
            print("#####detect more one:",source_img)
            print("the number of people : ", len(humans))
            #bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            #plt.imshow(bgimg)
            #plt.show()
            '''

        logging___ += "the number of people in " + trainname + " is " + str(len(keypoints))+"\n"
        
        for i in range(len(keypoints)):# number of human
#            annotations=[]
            num_keypoints= 0
            k_list= np.zeros(54,dtype='int').tolist()
            k = keypoints[i]#single human

            for j in range(18):#k.keys()=18
                if j not in k.keys():
                    continue
                num_keypoints +=1
            
                (x,y) = k[j]
                k_list[insert_[j] : insert_[j] +3] = [k[j][0],k[j][1],2]
            # print("=====")
            # print(k_list)
            # print("=====")
            
            annotation =  {'segmentation': [],
             'num_keypoints': num_keypoints,
             'area': 0,
             'iscrowd': 0,
             'keypoints': k_list,
             'image_id': image_id,
             'bbox': [],
             'category_id': 1,
             'id': idcount}
         
            annotations.append(annotation)
            #annotations.insert(0,annotation)
            idcount+=1
        #del humans,keypoints,image,original_image
        

    myjson={
        "info": info,
        "licenses": [licenses],
        "images": images,
        "annotations": annotations,
        "categories": [categories]
        }
    
    import json
    
    with open( r"C:\Users\adm\Desktop\11_03\cocoimg1__.json" , "w" ) as f:
        json.dump(myjson,f)
        print ( " ok... " )
    
    loggingimpfor = "the number of img : " +str(len(allfile))+"\n"
    loggingimpfor+= "totla detect keypoints" + str(idcount)
    with open(r"C:\Users\adm\Desktop\11_03\logging"+filepath[-4] +"__.txt","w") as f_:
        f_.write(logging___ + loggingimpfor)


    print("the number of img : " , len(allfile))
    print("totla detect number",idcount)
    #print("testtotal is ",testtotal)

############################################################################################################################


    '''
    import matplotlib.pyplot as plt

    fig = plt.figure()

    
    the keypoints : [{0: (252, 126), 1: (252, 172), 2: (216, 170), 3: (204, 218), 
    4: (194, 214), 5: (284, 166), 6: (320, 224), 7: (276, 288), 8: (224, 300), 
    11: (260, 302), 12: (282, 360), 14: (242, 116), 15: (264, 116), 16: (230, 122), 
    17: (282, 120)}] 
    
    
    #a = fig.add_subplot(2, 2, 1)
    #a.set_title('Result')
    #img_ = cv2.imread('./images/apink1.jpg')
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #cv2.circle(img_, (82,259), 10, (255,0,0), thickness=3, lineType=8, shift=0)
    #plt.imshow(img_)
    plt.show()
    
    
    '''
    
    '''
    ## human = BodyPart( uidx, part_idx, x, y, score)
    annotation={}
    annotation['iscrowd'] = 0
    annotation['keypoints']=[]
    #the format is x,y,v
    #v=1: no annotate , v = 2:annotate but unvisiable , v=3 is ok 
    
    try to get point
    for human in humans:
        annotation['num_keypoints'] = len(human.body_parts.keys())
        
        if ( len(humans) > 1):annotation['iscrowd'] = 1
        
        for i in range(18):
            
            if(i in human.body_parts.keys()):
            
                annotation['keypoints'].append(human.body_parts[i][2])
                annotation['keypoints'].append(human.body_parts[i][3] )
                annotation['keypoints'].append(1)
            else: continue
            
    
    '''


