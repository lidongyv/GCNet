# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:15:18 2017

@author: lidong
"""

import os

import numpy as np
import tensorflow as tf
from python_pfm import *
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
import cv2
gfilenames=gfile.Glob(os.path.join(r'D:\stereo dataset\Stereo Matching\disparity','*.pfm'))
a=readPFM(gfilenames[1])
b=np.floor(a[0])
win = cv2.namedWindow('test win', flags=0)
cv2.imshow('test win', b.astype(np.uint8))
cv2.waitKey(0)  