# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 21:21:40 2017

@author: lidong
"""
from tensorflow.python.platform import gfile
import os
path=r'D:\stereo dataset\Stereo Matching\train_data\data\1'
ifilenames=gfile.Glob(os.path.join(path,'*.png'))
data_path=r'D:\stereo dataset\Stereo Matching\train_data'
ilfilenames=gfile.Glob(os.path.join(data_path,r'data\1','*left','*.png'))
irfilenames=gfile.Glob(os.path.join(data_path,r'data\1','*right','*.png'))