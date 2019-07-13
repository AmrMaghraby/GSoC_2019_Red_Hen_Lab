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
            g[i] = slim.conv2d(h[i],num_outputs[i],3)
          
        F_Score = slim.conv2d(g[3],1,1,activation_fn=tf.nn.sigmoid,normalizer_fn=None)
        geo_map = slim.conv2d(g[3],4,1,activation_fn=tf.nn.sigmoid,normalizer_fn=None) * FLAGS.text_scale
        angle_map = (slim.conv2d(g[3],1,1,activation_fn = tf.nn.sigmoid,normalizer_fn=None) - 0.5) * np.pi/2
        F_geometry = tf.concat([geo_map, angle_map],axis=-1)
    
    return g[3], F_score, F_geometry  
  
  def dice_coefficient(self, y_true_cis, y_pred_cls, training_mask):
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1 - (2 * intersection / union)
    return loss
  
  def loss(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
    classification_loss = self.dice_coefficient(y_true_cls,y_pred_cls,training_mask)
    classification_loss *= 0.01
    #d1 -> top d2 -> right d3 -> bottom d4 -> left
    d1,d2,d3,d4,theta = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred,d2_pred,d3_pred,d4_pred,theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1 + d3) * (d2 + d4)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2,d2_pred) + tf.minimum(d4,d4_pred)
    h_union = tf.minimum(d1,d1_pred) + tf.minimum(d3,d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    L_g = L_AABB + 20 * L_theta
    
    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
    
