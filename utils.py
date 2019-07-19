# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:30:01 2018
edited by Lin_Shien

reference page : https://github.com/fandulu
author : Fan Yang
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import glob
import scipy.ndimage.interpolation as inter


class SBU_dataset():
    def __init__(self, dir):
        print ('loading data from:', dir)
        self.SBU_dir = dir
        self.pose_paths = glob.glob(os.path.join(self.SBU_dir, 's*', '*','*','*.txt'))  # 282 paths
        self.pose_paths.sort()
                   
    def get_data2D(self, test_set_folder):                # test_set_folder 用來當作 testing data
        '''
        SBU dataset has 282 skeleton sequence of 8 classes
        '''
        cross_set = {}
        cross_set[0] = ['s01s02', 's03s04', 's05s02', 's06s04']
        cross_set[1] = ['ds02s03', 's02s07', 's03s05', 's05s03']
        cross_set[2] = ['s01s03', 's01s07', 's07s01', 's07s03']
        cross_set[3] = ['s02s01', 's02s06', 's03s02', 's03s06']
        cross_set[4] = ['s04s02', 's04s03', 's04s06', 's06s02', 's06s03']
        
        def read_txt(pose_path):
            banned_idx = [ 2 + 3 * idx for idx in range(30)]
            coord_2D = []
            a = pd.read_csv(pose_path, header = None).T    # (91, frames)
            a = a[1:]            # (90, frames)
            a = a.as_matrix()    # (90, frames)
            
            for idx in range(90) :
                if not (idx in banned_idx) :
                    coord_2D.append(a[idx])
                    
            matrix_2D = np.array(coord_2D)     # (60, frames)
            
            return matrix_2D
        
        print('test set folder should be slected from 0 ~ 4')
        print('selected test folder {} includes:'.format(test_set_folder), cross_set[test_set_folder])  

        train_set = []
        test_set = []
        for i in range(len(cross_set)):    # 選 testing data 為哪一個 dict of file names
            if i == test_set_folder:
                test_set += cross_set[i]
            else:
                train_set += cross_set[i]

        train = {} 
        test = {}

        for i in range(1,9):    # 8 classes
            train[i] = []
            test[i] = []

        for pose_path in self.pose_paths:
            pose = read_txt(pose_path)
            if pose_path.split('\\')[-4] in train_set:   
                train[int(pose_path.split('\\')[-3])].append(pose)   # a (90, frames)
            else:
                test[int(pose_path.split('\\')[-3])].append(pose) 
        
        # then convert data to maxtrix form
        X_0, X_1, X_2, X_3, Y_SBU = self.data_to_matrix2D(train)
        X_TEST_0, X_TEST_1, X_TEST_2, X_TEST_3, Y_TEST = self.data_to_matrix2D(test)  
        
        return {'train' : [X_0, X_1, X_2, X_3, Y_SBU], 'test' : [X_TEST_0, X_TEST_1, X_TEST_2, X_TEST_3, Y_TEST]}
    
    def data_to_matrix2D(self, data_dict) :        
        X_0 = []
        X_1 = []
        X_2 = []
        X_3 = []
        Y = []

        for i in range(1,9):                              # loop 8 classes
            for j in range(len(data_dict[i])):            # loop all samples within the same class
                
                #First person pose
                p_0 = np.copy(data_dict[i][j].T[:,:30])   # (frames, 30)
                p_0 = p_0.reshape([-1,15,2])              # (frames, joints, 2)
                t_0 = p_0.shape[0]                        # the number of all frames
                if t_0 > 20:                              # sample the range from crop size of [16,t_0]
                    ratio = np.random.uniform(1, t_0 / 20)   
                    l = int(20 * ratio)
                    start = random.sample(range(t_0 - l), 1)[0]
                    end = start + l
                    p_0 = p_0[start:end,:,:]
                    p_0 = zoom2D(p_0)
                elif t_0 < 20:
                        p_0 = zoom2D(p_0)

                #Second person pose
                p_1 = np.copy(data_dict[i][j].T[:,30:])
                p_1 = p_1.reshape([-1,15,2])
                t_1 = p_1.shape[0]
                if t_1 >20:  
                    ratio = np.random.uniform(1,t_1/20)
                    l = int(20 * ratio)                          # ratio 後的 length
                    start = random.sample(range(t_1-l),1)[0]   # 抽樣找到起始點
                    end = start + l                             
                    p_1 = p_1[start:end,:,:]
                    p_1 = zoom2D(p_1)
                elif t_1 < 20:
                    p_1 = zoom2D(p_1)
            
                # randomly mirror augmentation 
                # since two persions' postion could be switched
                if np.random.choice([0,1],1): 
                    p_0, p_1 = mirror(p_0, p_1)
                    
                    #Calculate the temporal difference
                p_0_diff = p_0[1:,:,:]-p_0[:-1,:,:]
                p_0_diff = np.concatenate((p_0_diff,np.expand_dims(p_0_diff[-1,:,:],axis=0)))
                p_1_diff = p_1[1:,:,:]-p_1[:-1,:,:]
                p_1_diff = np.concatenate((p_1_diff,np.expand_dims(p_1_diff[-1,:,:],axis=0)))
                    
                X_0.append(p_0)
                X_1.append(p_0_diff)
                X_2.append(p_1)
                X_3.append(p_1_diff)
                    
                label = np.zeros(10)
                label[i-1] = 1
                Y.append(label)

        X_0 = np.stack(X_0)
        X_1 = np.stack(X_1)
        X_2 = np.stack(X_2)
        X_3 = np.stack(X_3)
        Y = np.stack(Y)
    
        return X_0, X_1, X_2, X_3, Y
 

class my_dataset():
    def __init__(self):
        pass
    
    def read_txt(self, pos_path):
        a = pd.read_csv(pos_path, header = None).T
        return a.as_matrix()
    
    def data_to_fixedFrame(self, pos_path, oh) :
        mat = self.read_txt(pos_path)
        train_set = [] 
        train_label = []
        for idx in range(int(mat.shape[1] / 50)) :
            train_set.append(mat[:, idx * 50 : (idx + 1) * 50])
    
        train_set.append(mat[:, int(mat.shape[1] / 50) * 50 :])
    
        for idx in range(len(train_set)) :
            label = np.zeros(10)
            label[oh-1] = 1
            train_label.append(label)
  
        return train_set, train_label
    
    def data_to_matrix2D(self, train_set, train_label) :
        X = []
        X_diff = []
    
        for idx in range(len(train_set)):            # loop all samples within the same class                
            #First person pose
            p_0 = np.copy(train_set[idx].T)             # (frames, 30)
            p_0 = p_0.reshape([-1, 15, 2])              # (frames, joints, 2)
            t_0 = p_0.shape[0]
            if t_0 > 20:                                # sample the range from crop size of [16,t_0]
                ratio = np.random.uniform(1, t_0 / 20)   
                l = int(20*ratio)
                start = random.sample(range(t_0 - l), 1)[0]
                end = start + l
                p_0 = p_0[start:end,:,:]
                p_0 = zoom2D(p_0)
            elif t_0 < 20:
                p_0 = zoom2D(p_0)   
                            
            #Calculate the temporal difference
            p_0_diff = p_0[1:,:,:]-p_0[:-1,:,:]
            p_0_diff = np.concatenate((p_0_diff,np.expand_dims(p_0_diff[-1,:,:],axis=0)))

            X.append(p_0)
            X_diff.append(p_0_diff)
    
        X = np.stack(X)
        X_diff = np.stack(X_diff)
        Y = np.stack(train_label)
    
        return X, X_diff, Y
    
    def get_data2D(self):
        d1, l1 = self.data_to_fixedFrame(r"C:\Users\Lin_Shien\Desktop\fall_detection\data_seq\fall\cocoimg2\skeleton_pos.txt", 9)
        d2, l2 = self.data_to_fixedFrame(r"C:\Users\Lin_Shien\Desktop\fall_detection\data_seq\sit\cocoimg11\skeleton_pos.txt", 10)
        d3, l3 = self.data_to_fixedFrame(r"C:\Users\Lin_Shien\Desktop\fall_detection\data_seq\fall\cocoimg2\skeleton_pos.txt", 9)
        #d4, l4 = data_to_fixedFrame(r"C:\Users\Lin_Shien\Desktop\fall_detection\data_seq\walk\cocoimg9\skeleton_pos.txt", 10)
        
        X1, X1_diff, Y1 = self.data_to_matrix2D(d1, l1)
        X2, X2_diff, Y2 = self.data_to_matrix2D(d2, l2)
        X3, X3_diff, Y3 = self.data_to_matrix2D(d3, l3)
        #X4, X4_diff, Y4 = data_to_matrix(d4, l4)
        
        X = np.concatenate((X1[:25], X2[:50], X3[:25]))
        X_diff = np.concatenate((X1_diff[:25], X2_diff[:50], X3_diff[:25]))
        Y = np.concatenate((Y1[:25], Y2[:50], Y3[:25]))

        T = np.concatenate((X1[25:], X2[50:], X3[25:]))
        T_diff = np.concatenate((X1_diff[25:], X2_diff[50:], X3_diff[25:]))
        TY = np.concatenate((Y1[25:], Y2[50:], Y3[25:]))
        
        return {'train' : [X, X_diff, Y], 'test' : [T, T_diff, TY]}
        
        
#Transfer to orginial coordinates for plotting
def coord2org(p): 
    p_new = np.empty_like(p)
    for i in range(15):
        p_new[i,0] = 640 - (p[i,0] * 640)
        p_new[i,1] = 480 - (p[i,1] * 240)
    return p_new

#Plotting the pose
def draw_2d_pose(gtorigs): 
    f_ind = np.array([
        [2,1,0],
        [3,6,2,3],
        [3,4,5],
        [6,7,8],
        [2,12,13,14],
        [2,9,10,11],      
    ])

    fig = plt.figure()
    
    axes = plt.gca()
    axes.set_xlim([0,640])
    axes.set_ylim([0,480])

    ax = fig.add_subplot(111)
    
    for gtorig,color in zip(gtorigs,['r','b']):
        
        gtorig = coord2org(gtorig)
        
        for i in range(f_ind.shape[0]):
        
            ax.plot(gtorig[f_ind[i], 0], gtorig[f_ind[i], 1], c=color)
            ax.scatter(gtorig[f_ind[i], 0], gtorig[f_ind[i], 1],s=10,c=color)
        
    plt.show()

#Rescale to be 20 frames

def zoom2D(p):
    l = p.shape[0]                   # frames before normalizaion
    p_new = np.empty([20,15,2])      
    for m in range(15):
        for n in range(2):
            p_new[:,m,n] = inter.zoom(p[:,m,n], 20/l)[:20]   # 把 joint coord 作縮放
    return p_new

#Switch two persons' position
def mirror(p_0,p_1):
    p_0_new = np.copy(p_0)
    p_1_new = np.copy(p_1)
    p_0_new[:,:,0] = abs(p_0_new[:,:,0]-1)
    p_1_new[:,:,0] = abs(p_1_new[:,:,0]-1)
    return p_0_new, p_1_new
