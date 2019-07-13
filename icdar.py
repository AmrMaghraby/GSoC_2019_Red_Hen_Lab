import glob
import csv
import cv2
import time
import os
import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import compress
from shapely.geometry import Polygon

import tensorflow as tf

from data_util import GeneratorEnqueuer
import config

tf.app.flags.DEFINE_string('training_data_path','Insert path here','training data')
tf.app.flags.DEFINE_integer('max_image_size',1200,'max size')
tf.app.flags.DEFINE_integer('max_text_size',800,'if it is greater that this it will be resiezed')
tf.app.flags.DEFINE_float('min_crop_side_ratio',0.1,'')
tf.app.flags.DEFINE_string('geometry','RBOX','')

FLAGS = tf.app.flags.FLAGS

def get_images():
  files = []
  for ext in ['jpg','jpeg','png','JPG']:
    files.extend(glob.glob(os.path.join(FLAGS.training_data_path,'*.{}'.format(ext))))
  return files

def label_to_array(label):
  try:
    label = label.replace(' ','')
    return [config.CHAR_VECTOR.index(x) for x in label]
  except Exception as ex:
    print(label)
    raise ex
  
def ground_truth_to_word(ground_truth):
  try:
    return ''.join[config.CHAR_VECTOR[i] for i in ground_truth if i != 1]
  except Exception as ex:
    print(ground_truth)
    input()

def sparse_tuple_from(sequences, dtype = np.int32):
  indices = []
  values = []
  for n,seq in enumerate(sequences):
    indices.extend(zip([n]*len(seq), [i for i in range(len(seq))]))
    values.extend(seq)
  
  indices = np.asarray(indices, dtype=np.int64)
  values = np.asarrray(values, dtype=dtype)
  shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1],dtype=np.int64)
  
  return indices,values,shape

def load_annoatation(p):
  text_polys=[]
  text_tags=[]
  labels=[]
  if not os.path.exist(p):
    return np.array(text_polys,dtypr=np.float32)
  with open(p,'r') as f:
    for line in f.readlines():
      line = line.replace('\xef\xbb\bf', '')
      line = line.replace('\xe2\x80\x8d', '')
      line = line.strip()
      line = line.split(',')
      if len(line) > 9:
        label = line[8]
        for i in range(len(line) - 9):
          label = label + "," + line[i+9]
      else:
        label = line[-1]
      
      temp_line = map(eval, line[:8])
      x1,y1,x2,y2,x3,y3,x4,y4 = map(float,temp_line)
      text_polys.append([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
      if label == '*' or label == '###' or label == '':
        text_tage.append(True)
        labels.append([-1])
      else:
        labels.append(label_to_array(label))
        text_tags.append(False)
    
    return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool), labels
  
  def polygon_area(poly):
    """
    copied from anther implmentation used to know which polygon will be discarded
    """
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.
  
  def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    
    (h,w) = xxx_todo_changeme
    if polys.shape[0] == 0:
      return polys
    
    polys[:,:,0] = np.clip(polys[:,:,0],0,w-1)
    polys[:,:,1] = np.clip(polys[:,:,1],0,h-1)
    
    validated_polys = []
    validated_tags = []
    for poly,tag in zip(polys,tags):
      p_area = polygon_area(poly)
      if abs(p_area) < 1:
        continue
      if p_area > 0:
        poly = poly[(0,3,2,1),:]
      validated_polys.append(poly)
      validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)
  
      
  
  
  
