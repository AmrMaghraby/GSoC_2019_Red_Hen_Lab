import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale',512,'')

from nets import resnet_v1

FLAGS = tf.app.flags.FLAGS

def unpool(inputs):
  return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2, tf.shape(inputs)[2]*2])

def mean_image_subtraction(images, means=[123.68,116.78,103.94]):
  num_channels = images.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError(' Len dont match each other')
  channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=3, values=channels)

class Backbone(object):
  def __init__(self,is_training=True):
    self.is_training = is_training
    
  def model(self,image,wieght_decay=1e-5):
    
    images = mean_image_subtraction(image)
    
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
      logits,endpoints = resnet_v1.resnet_v1_50(images,is_training = self.is_training,scope='resnet_v1_50')
      
    with tf.variable_scope('feature_fusion',values=[end_points.values]):
      batch_norm_params = {
      'decay': 0.997,
      'epsilon': 1e-5,
      'scale': True,
      'is_Training': self.is_training
      }
      with slim.arg_scope([slim.conv2d], 
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=batch_norm_params,
                          weights_regularizer=slim.l2_regularizer(weight_decay)):
      
        f = [endpoints['pool5'],endpoints['pool4'],
             endpoints['pool3'],endpoints['pool2']]
        
        g = [None,None,None,None]
        h = [None,None,None,None]
        num_outputs = [None,128,64,32]
        
        for i in range(4):
          if i == 0:
            h[i] = f[i]
          else:
            c1 = slim.conv2d(tf.concat([g[i-1],f[i]], axis=-1), num_outputs[i],1)
            h[i] = slim.conv2d(c1,num_outputs[i],3)
          if i <= 2:
            g[i] = unpool(h[i])
          else:
            g[i] = slim.conv2(h[i],num_outputs[i],3)
          
        
        
        
        
