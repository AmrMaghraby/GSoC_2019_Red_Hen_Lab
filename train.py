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
    recognition_loss = recognition.loss(recognition_logits,transcription,BoxWidth)
    tf.summary.scalar('detect_loss',detection_loss)
    tf.summary.scalar('Recognition_loss',recognition_loss)
    return detection_loss,recognition_loss, detection_loss + lamda * recognition_loss

"""
Main function we could call it also driver function
"""
def main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MKDir(FLAGs.checkpoint_path)
    else:
        if not FLAGS.restore
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MKDir(FLAGS.checkpoint_path)
            
    Images = tf.placeholder(tf.float32,shape[None,None,None,3],name = 'Images')
    Score_maps = tf.placeholder(tf.float32,shape[None,None,None,1],name = 'score_maps')
    geo_maps = tf.placeholder(tf.float32,shape[None,None,None,5],name = 'geo_maps')
    training_masks = tf.placeholder(tf.float32,shape[None,None,None,1],name = 'training_mask')
    transcription = tf.sparse_placeholder(tf.int32,name='transcription')
    
    Transform_matrix = tf.placeholder(tf.float32,shape=[None,6],name='transform_matrix')
    BBox = []
    BoxWidth = tf.placeholder(tf.int32,shape=[None], name='BoxWidth')
    
    for i in range(FLAGS.batch_size_per_gpu):
        input_box_masks.append(tf.placeholder(tf.int32,shape=[None],name='BBox' + str(i)))
    
    f_score, f_geometry, recognition_logits, decode = build_graph(Images,Transform_matrix, BBox, BoxWidth)
    
    global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable = False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step,decay_steps=10000,decay_rate=0.94,staircase=True)
    tf.summary.scalar('learning_rate',learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    
    d_loss,r_loss,model_loss = compute_loss(f_score, f_geometry, recognition_logits, score_maps,  geo_maps, training_masks, transcription,BoxWidth)
    tf.summary.scalar('total_loss',model_loss)
    total_loss = model_loss
    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.update_OPS))
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    summary_op = tf.summary.merge_all()
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay,global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')
        
    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())
    
    init = tf.global_variables_initializer()
    
    if FLAGS.pretrained_model_path is not None:
        ckpt = tf.train.latest_checkpoint(FLAGS.pretrained_model_path)
        variable_restore_op = slim.assign_from_checkpoint_fn(ckpt, slim.get_trainable_variables(),ignore_missing_vars=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        if FLAGS.restore:
            print('Loading previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess,ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)
        
        data_generator = icdar.get_batch(num_workers = FLAGS.num_readers,
                                        input_size = FLAGS.input_size,
                                        batch_size = FLAGS.batch_size_per_gpu)
        
        start = time.time()
        for step in range(FLAGS.max_steps):
            data = next(data_generator)
            inp_dict = {Images: data[0],
                        score_maps: data[2],
                        geo_maps: data[3],
                        training_masks: data[4],
                        transform_matrix: data[5],
                        BoxWidth: data[7],
                        Transcription: data[8]}
            for i in range(FLAGS.batch_size_per_gpu):
                inp_dict[input_box_masks[i]] = data[6][i]
                
            
            dl,rl,tl,_ = sess.run([d_loss, r_loss, total_loss, train_op],feed_dict=inp_dict)
            if np.isnan(tl):
                print('Stopping condition')
                break
            
            if step % 10 == 0:
                avg_time_per_step = (time.time() - start)/10
                start = time.time()
                print('Step {:06d}, detect_loss {:.4f}, recognize_loss {:.4f}, total_loss {:.4f}, {:.2f} second/step'.format(step,dl,rl,tl,avg_time_per_step))
                
            if step % FLAGS.save_checkpoint == 0:
                saver.save(sess, FLAGS.checkpoint_path+'model.ckpt', global_step=global_step)
            
            if step % FLAGS.save_summary_steps == 0:
                summary_writer.add_summary(summary_str,global_step=step)
                
 if __name__ == '__main__':
    tf.app.run()
                
