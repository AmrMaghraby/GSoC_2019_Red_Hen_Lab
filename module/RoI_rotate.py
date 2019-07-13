import sys
sys.path.append("..")
import numpy as np
import cv2
import tensorflow as tf
import math
import config
import os

from stn import spatial_transformer_network as transformer

class RoIRotate(object):
        def __init__(self,height=8):
                self.height = height
        def roi_rotate_tensor(self,feature_map,tarnsform_matrix, box_masks,box_width,is_debug=false):
                max_width = box_widths[tf.argmax(0,box_widths,output_type = tf.int32)]
                box_width = tf.cast(box_width,tf.float32)
                tile_feature_maps = []
                
                for i in enumerate(box_masks):
                        _feature_map = feature_map[i]
                        _feature_mao = tf.expand_dims(_feature_map, axis=0)
                        box_numms = tf.shape(mask)[0]
                        _feature_map = tf.tile(_feature_map,[box_nums,1,1,1])
                        tile_feature_maps.append(_feature_map)
                        
                tile_feature_maps = tf.concat(tile_feature_maps,axis=0)
                norm_box_width = box_width / map_shape[2]
                ones = tf.ones_like(norm_box_width)
                norm_box_heights = ones * (8.0 / map_shape[1])
		zeros = tf.zeros_like(norm_box_widths)
		crop_boxes = tf.transpose(tf.stack([zeros, zeros, norm_box_heights, norm_box_widths]))
                crop_size = tf.transpose(tf.stack([8, max_width]))
                trans_feature_map = transformer(tile_feature_maps, transform_matrixs)
                
                box_inds = tf.range(tf.shape(trans_feature_map)[0])
		rois = tf.image.crop_and_resize(trans_feature_map, crop_boxes, box_inds, crop_size)
		pad_rois = tf.image.pad_to_bounding_box(rois, 0, 0, 8, max_width)
		print "pad_rois: ", pad_rois
		return pad_rois

                
