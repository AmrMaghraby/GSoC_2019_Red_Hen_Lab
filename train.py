import time
import numpy as np
import tensorflow.contrib import slim

#Define Flags to be used later

tf.app.flags.DEFINE_integer('input_size',512,'')
tf.app.flags.DEFINE_integer('batch_size_per_gpu',8,'')
tf.app.flags.DEFINE_integer('num_readers',8,'')
tf.app.flags.DEFINE_float('learning_rate',0.0001,'')
tf.app.flags.DEFINE_integer('max_steps',100000,'')
tf.app.flags.DEFINE_float('moving_average_decay','0.997','')
tf.app.flags.DEFINE_string('gpu_list','1','')
tf.app.flags.DEFINE_string('checkpoint_path','checkpoints/','')
tf.app.flags.DEFINE_boolean('restore',False,'')
tf.app.flags.DEFINE_integer('save_checkpoint_steps',1000,'')
tf.app.flags.DEFINE_integer('save_summary_steps',100,'')
tf.app.flags.DEFINE_string('pretrianed_model_path','sythn_pretrained_model/','')

import icdar
from module import SharedConv,Recognition,RoI_rotate

FLAGS = tf.app.flags.FLAGS

Detection = SharedConv.Backbone(is_training=True)
Rotation = RoI_rotate.RoIRotate()
recognition = Recognition.Recognition(is_training=True)


"""
First Build the graph as discriped in paper FOTS Shared Conv then pass it to ROI rotate to detect text with Boundry
Box predicition.
we will use the output of ROI_rotate in recognition branch (CNN - BLSTM) and decoding it with CTC
"""
def build_graph(Images,Transform_matrix,BBox,BoxWidth):
    seq_len = BoxWidth[tf.argmax(BoxWidth,0)] * tf.ones_like(BoxWidth)
    shared_feature, f_score, f_geometry = Detection.model(Images)
    pad_rois = Rotation.roi_rotate_tensor(shared_feature, Transform_matrix, BBox, BoxWidth)
    recognition_logits = recognition.build_graph(pad_rois, seq_len)
    _, dense_decode = recognition.decode(recognition_logits, seq_len)
    return f_scroe, f_geometry, recognition_logits, dense_decode
    
"""
Here we are going to define Compute Loss function to define how we are near from the real results.
Our Metrics : detection_Loss,Recognition_Loss.
"""
def compute_loss(f_score,f_geometry,recognition_logits,score_maps,geo_maps,training_masks,transcription,BoxWidth,lamda=0.01):    
    detection_loss = Detection.loss(score_maps,f_score,geo_maps,f_geometry,training_masks)
    recognition_loss = recognition.loss(recognition_logits)
    
