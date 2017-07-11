# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:17:51 2017

@author: lidong
"""
import tensorflow as tf
import time
import numpy as np
# Creates a graph.
"""
def _todisparity(volume):
    with tf.device('/gpu:1'):
        t2=time.time()
        d=np.arange(192,dtype=np.float32)
        d=np.reshape(d,[192,1,1])  
        d=tf.convert_to_tensor(d)
        disparities=tf.reshape(volume,[192,540,960])
        probability=tf.nn.softmax(-disparities,dim=0)
        #print('time:'+str(time.time()-t2))
        sums=tf.reduce_sum(d*probability,0)
        #print('time:'+str(time.time()-t2))
        #t2=time.time()
        
        print('time:'+str(time.time()-t2))       
    return sums
t1=time.time()
c=tf.constant([100,100],1)
volume=tf.zeros([1,100,100,100,1])
disparity=tf.ones([192,540,960])
#a=_softargmin(disparity)
b=_todisparity(disparity)
#disparity=tf.range(1,1000000,1,dtype=tf.float32)
#probability=tf.nn.softmax(disparity)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    print(time.time()-t1)
    sess.run(init_op)
    print(time.time()-t1)
    print(b.shape)
    print(time.time()-t1)
"""

"""
# Creates a session with log_device_placement set to True.
tf.device('/gpu:0')
volume=tf.zeros([1,96,270,480,64],dtype=tf.float16)

#c=tf.zeros([1,192/2,540/2,960/2,64])
#d=2*volume
with tf.device('/gpu:0'):
    lvolume=[]
    single_dis=np.zeros([1,270,480,64],dtype=np.float32)
    single_dis=tf.convert_to_tensor(single_dis)
    for i in range(50):
        #b=tf.split(single_dis,[240,240],2)
        #a=tf.concat([b[0],b[1]],0)
        lvolume.append(single_dis)
    lvol=tf.stack(lvolume,axis=0)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(lvol))


lvolume=[]
for i in range(96):
    single_dis=np.zeros([1,270,480,64],dtype=np.float32)
    lvolume.append(single_dis)
print(lvolume)
""" 
tf.device('/gpu:0')
lvolume=[]
single_dis=np.zeros([1,128,256,64],dtype=np.float32)
single_dis=tf.convert_to_tensor(single_dis)
for i in range(96):
	splits=tf.split(single_dis,[256-i,i],axis=2)
	rsplits=tf.concat([splits[1],splits[0]],axis=2)
	single_dis2=tf.concat([single_dis,rsplits],3)
	lvolume.append(single_dis2)
lvol=tf.stack(lvolume,axis=0)
rvolume=[]
for i in range(96):
	splits=tf.split(single_dis,[256-i,i],axis=2)
	rsplits=tf.concat([splits[1],splits[0]],axis=2)
	single_dis2=tf.concat([single_dis,rsplits],3)
	rvolume.append(single_dis2)
rvol=tf.stack(lvolume,axis=0)  
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(lvol)) 
sess.run(rvol)

