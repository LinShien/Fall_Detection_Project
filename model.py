from __future__ import division
from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras import backend as K
from keras.optimizers import rmsprop
import tensorflow as tf

def one_obj(frame_l = 16, joint_n = 15, joint_d = 2):
    '''
    frame_l : frame 長度
    joint_n : # of keypoint or joint
    joint_d : axis number
    '''
    input_joints = Input(name = 'joints', shape = (frame_l, joint_n, joint_d))              # spatial stream
    input_joints_diff = Input(name = 'joints_diff', shape = (frame_l, joint_n, joint_d))    # temporal stream
    
    ##########branch 1##############
    x = Conv2D(filters = 32, kernel_size=(1, 1), padding='same')(input_joints)      # conv1, (?, frames, joints, 32)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(filters = 16, kernel_size = (3, 1), padding='same')(x)               # conv2, (?, frames, joints, 16)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Permute((1, 3, 2))(x)        # transpose
    
    x = Conv2D(filters = 16, kernel_size=(3,3), padding='same')(x)                  # conv3, (?, frames, joints, 16)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)   
    ##########branch 1##############
    
    ##########branch 2##############Temporal difference
    x_d = Conv2D(filters = 32, kernel_size = (1, 1), padding='same')(input_joints_diff)   # conv1
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)
    
    x_d = Conv2D(filters = 16, kernel_size = (3, 1),padding = 'same')(x_d)                # conv2
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)

    x_d = Permute((1,3,2))(x_d)
    
    x_d = Conv2D(filters = 16, kernel_size = (3, 3),padding = 'same')(x_d)                # conv3
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)
    ##########branch 2##############
    
    x = concatenate([x, x_d], axis = -1)           # concatenate the spatial and temporal feature maps, (?, frames, joints, 32)
    
    x = Conv2D(filters = 32, kernel_size = (1, 1), padding = 'same')(x)        # conv5, 可接上 temporal proposal, (?, frames, joints, 32)
    
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x) 
    x = Dropout(0.1)(x)
       
    x = Conv2D(filters = 64, kernel_size = (1, 1), padding = 'same')(x)        # conv6, 可接上 temporal proposal, (?, frames, joints, 32)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x) 
    x = Dropout(0.1)(x)
    
    
    model = Model([input_joints, input_joints_diff], x)

    return model

def multi_obj(frame_l = 16, joint_n = 15, joint_d = 2):
    inp_j_0 = Input(name = 'inp_j_0', shape = (frame_l, joint_n, joint_d))
    inp_j_diff_0 = Input(name = 'inp_j_diff_0', shape = (frame_l, joint_n, joint_d))
    
    inp_j_1 = Input(name = 'inp_j_1', shape = (frame_l, joint_n, joint_d))
    inp_j_diff_1 = Input(name = 'inp_j_diff_1', shape = (frame_l, joint_n, joint_d))
    
    single = one_obj(joint_d = joint_d)
    x_0 = single([inp_j_0, inp_j_diff_0])   # person 1's conv6 output
    x_1 = single([inp_j_1, inp_j_diff_1])   # person 2's conv6 output
      
    x = Maximum()([x_0, x_1])  # 提取每個人動作的最大值
    '''
    dim_ls = x.get_shape().as_list()
    x_fla = Reshape((dim_ls[1], dim_ls[2] *dim_ls[3]))(x)
    x = Dense(20, activation = "relu")(x_fla)
    x = Dense(1, activation = "relu")(x)
    alphas = Activation(activation = 'softmax', name='attention_weights')(x)
    x = Dot(axes = 1)([alphas, x_fla])
    '''
    
    x = Flatten()(x)
    x = Dropout(0.1)(x)
     
    x = Dense(256)(x)             # FC layer
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    
    x = Dense(10, activation = 'softmax')(x)      # FC layer
    
    model = Model([inp_j_0, inp_j_diff_0, inp_j_1, inp_j_diff_1], x)   # 2 人的 model
    
    return model