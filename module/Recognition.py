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
    
    cnn_feature = self.cnn(rois)
    cnn_feature = tf.squeeze(cnn_feature,axis=-1)
    reshape_cnn_feature = cnn_feature
    lstm_output = self.bilstm(reshape_cnn_feature,seq_len)
    logits = tf.reshape(lstm_output,[-1,self.rnn_hidden_num * 2])
    W = tf.Variable(tf.truncated_normal([self.rnn_hidden_num * 2, self.num_classes], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0., shape=[self.num_classes]), name="b")
		logits = tf.matmul(logits, W) + b
		logits = tf.reshape(logits, [nums, -1, self.num_classes])
		logits = tf.transpose(logits, (1, 0, 2))
		return logits
  
  def loss(self,logits,targets,seq_len):
    
    loss = tf.nn.CTC_loss(targets, logits, seq_len)
    recognition_loss = tf.reduce_mean(loss)
    return recognition_loss
    
    
  
