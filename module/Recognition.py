import tensorflow as tf
from tensorflow.contrib import slim, rnn
import numpy as np
import config
import os


class Recognition(object):
  def __init__(self,rnn_hidden_num=256,is_training=True):
    self.rnn_hidden_num = rnn_hidden_num
    self.is_training = is_training
    self.batch_norm_params = {'decay':0.997, 'epsilon':1e-5, 'scale':True,'is_training': is_training}
    self.num_classes = config.NUM_CLASSES
  
  def cnn(self,rois):
    with tf.variable_scope("recog/cnn"):
      conv1 = slim.conv2d(rois,64,3,stride=1,padding='SAME',activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm, normalizer_params=self.batch_norm_params)
      conv1 = slim.conv2d(conv1,64,3,stride=1,padding='SAME',activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm, normalizer_params=self.batch_norm_params)
      pool1 = slim.max_pool2d(conv1,[2,1],stride = [2,1])
      conv2 = slim.conv2d(pool1,128,3,stride=1,padding='SAME',activation_fn = tf.nn.relu,normalizer_fn=slim.batch_norm,normalizer_params=self.batch_norm_params)
      conv2 = slim.conv2d(conv2,128,3,stride=1,padding='SAME',activation_fn = tf.nn.relu,normalizer_fn=slim.batch_norm,normalizer_params=self.batch_norm_params)
      pool2 = slim.max_pool2d(conv2,[2,1],stride=[2,1])
      conv3 = slim.conv2d(pool2,256,3,stride=1,padding='SAME',activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,normalizer_params=self.batch_norm_params)
      conv3 = slim.conv2d(conv3,256,3,stride=1,padding='SAME',activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,normalizer_params=self.batch_norm_params)
      pool3 = slim.max_pool2d(conv3,[2,1],stride=[2,1])
    return pool3
  
  def bilstm(self,input_feature,seq_len):
    with tf.variable_scope("recog/rnn"):
      lstm_fw_cell = rnn.LSTMCell(self.rnn_hidden_num)
      lstm_bw_cell = rnn.LSTMCell(self.rnn_hidden_num)
      infer_output,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_feature, sequence_length = seq_len, dtype = tf.float32)
      infer_output = tf.concat(infer_output,axis = -1)
      return infer_output
    
  def build_graph(self,rois,seq_len,nums):
    
  
