# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:30:01 2018
edited by Lin_Shien

reference page : https://github.com/fandulu
author : Fan Yang
"""
import os
import numpy as np
from keras import optimizers
from utils import *
from model import *

# get SBU dataset data
SBU_dir = r'C:\Users\Lin_Shien\Desktop\fall_detection\data_seq\SBU'
sbu_dataset = SBU_dataset(SBU_dir)
sbu_data_pair = sbu_dataset.get_data2D(3)                        # selected from 0,1,2,3,4
X_0, X_1, X_2, X_3, Y_SBU = sbu_data_pair['train']
X_TEST_0, X_TEST_1, X_TEST_2, X_TEST_3, Y_TEST = sbu_data_pair['test']

# get addtional data
my_dataset = my_dataset()
my_data_pair = my_dataset.get_data2D()
X, X_diff, Y = my_data_pair['train']
T, T_diff, TY = my_data_pair['test']

X_train = np.concatenate((X_0, X))
X_train = np.concatenate((X_train, np.zeros(X.shape)))
X1_train = np.concatenate((X_1, X_diff))
X1_train = np.concatenate((X1_train, np.zeros(X_diff.shape)))

X2_train = np.concatenate((X_2, np.zeros(X.shape)))
X2_train = np.concatenate((X2_train, X))
X3_train = np.concatenate((X_3, np.zeros(X_diff.shape)))
X3_train = np.concatenate((X3_train, X_diff))
Y_train = np.concatenate((Y_SBU, Y, Y))

X_test = np.concatenate((X_TEST_0, T))
X1_test = np.concatenate((X_TEST_1, T_diff))
X2_test = np.concatenate((X_TEST_2, np.zeros(T.shape)))
X3_test = np.concatenate((X_TEST_3, np.zeros(T_diff.shape)))
Y_test = np.concatenate((Y_TEST, TY))

# CNN parameters
epochs = 400
lr = 0.001
adam = optimizers.Adam(lr)
model = multi_obj(frame_l = 20, joint_n = 15, joint_d = 2)
model.compile(adam, metrics = ['accuracy'], loss = 'mean_squared_error')
model.summary()

history = model.fit([X_train, X1_train, X2_train, X3_train], Y_train, batch_size = 32, epochs = epochs, verbose = True, shuffle = True)

Y_pred = model.predict([X_test, X1_test, X2_test, X3_test])

print('Predict labels:',np.argmax(Y_pred, axis=1))
print('Ground truth labels:', np.argmax(Y_test, axis=1))

loss, acc = model.evaluate([X_test, X1_test, X2_test, X3_test], Y_test, verbose = True)
print('testing loss : ', loss)
print('testing acc : ', acc)

