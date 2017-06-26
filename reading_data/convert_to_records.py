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
  num_examples = len(images)
  names=data_set['name']

  rows = images[1].shape[0]
  cols = images[1].shape[1]
  depth = images[1].shape[2]

  filename = os.path.join(r'D:\stereo dataset\Stereo Matching\testpfm', name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    label_raw=labels[index].tostring()
    name_raw=np.array(names[index]).tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label_raw':  _bytes_feature(label_raw),
        'name_raw':  _bytes_feature(name_raw),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()

def read_data_sets(data_path):
    ifilenames=gfile.Glob(os.path.join(data_path,'*.png'))
    gfilenames=gfile.Glob(os.path.join(data_path,'*.pfm'))
    image,disparity=[],[]
    for i in range(len(ifilenames)):
        image.append(cv2.imread(ifilenames[i]))
    for i in range(len(ifilenames)):
        disparity.append(readPFM(gfilenames[i]))
    return {'data':image,'label':disparity,'name':ifilenames}



data_sets=read_data_sets(r'D:\stereo dataset\Stereo Matching\testpfm')
  # Convert to Examples and write the result to TFRecords.
convert_to(data_sets, 'test')
  #convert_to(data_sets.validation, 'validation')
  #convert_to(data_sets.test, 'test')

