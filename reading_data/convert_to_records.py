# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from python_pfm import *
import tensorflow as tf
import cv2
from tensorflow.python.platform import gfile


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set['data']
  labels = data_set['label']
  num_examples = len(images[0])
  names=data_set['name']

  rows = images[0][0].shape[0]
  cols = images[0][0].shape[1]
  depth = images[0][0].shape[2]

  filename = os.path.join(r'D:\SceneFlow', name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_left_raw = images[0][index].tostring()
    image_right_raw=images[1][index].tostring()
    label_left_raw=labels[0][index].tostring()
    label_right_raw=labels[1][index].tostring()
    name_raw=np.array(names[index]).tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_left_raw':  _bytes_feature(image_left_raw),
        'image_right_raw':  _bytes_feature(image_right_raw),
        'label_left_raw':  _bytes_feature(label_left_raw),
        'label_right_raw':  _bytes_feature(label_right_raw),
        'name_raw':  _bytes_feature(name_raw)}))
    writer.write(example.SerializeToString())
  writer.close()

def read_data_sets(data_path):
    ilfilenames=gfile.Glob(os.path.join(data_path,r'data','*left','*.png'))
    irfilenames=gfile.Glob(os.path.join(data_path,r'data','*right','*.png'))
    glfilenames=gfile.Glob(os.path.join(data_path,r'groundtruth','*left','*.pfm'))
    grfilenames=gfile.Glob(os.path.join(data_path,r'groundtruth','*right','*.pfm'))
    image,disparity=[[],[]],[[],[]]
    for i in range(len(ilfilenames)):
        image[0].append(cv2.imread(ilfilenames[i]))
        image[1].append(cv2.imread(irfilenames[i]))
    for i in range(len(ilfilenames)):
        disparity[0].append(readPFM(glfilenames[i]))
        disparity[1].append(readPFM(grfilenames[i]))
    return {'data':image,'label':disparity,'name':ilfilenames}


print('C3')
data_sets=read_data_sets(r'D:\stereo dataset\Stereo Matching\tmp_data')
  # Convert to Examples and write the result to TFRecords.
convert_to(data_sets, 'tmp_data')
  #convert_to(data_sets.validation, 'validation')
  #convert_to(data_sets.test, 'test')

