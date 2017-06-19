# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 17:33:47 2017

@author: lidong
"""

import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
import cv2

FLAGS=flags.FLAGS

# Original image dimensions
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512
COLOR_CHAN = 3

# Default image dimensions.
IMG_WIDTH = 64
IMG_HEIGHT = 64

def get_input(mode=0):
	
