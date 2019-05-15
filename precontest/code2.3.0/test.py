#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:01:00 2019

@author: xudachengthu

define the method of test

/Users/xudachengthu/Downloads/GHdataset/ftraining-0.h5 --the file downloaded from crowdAI

uisng the .tfrecords file generated by generate.py
"""

import time
import tensorflow as tf
import numpy as np
import forward
import backward
import generate
TEST_INTERVAL_SECS = 10
TEST_NUM = 500

def test():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        #y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        y = forward.forwardpro(x, None)
        '''
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        '''
        saver = tf.train.Saver()
        
        wf_batch, pet_batch = generate.get_tfrecord(TEST_NUM, isTrain=False)
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    
                    xs, ys = sess.run([wf_batch, pet_batch])
                    
                    y_value = sess.run(y, feed_dict={x: xs})
                    y_c = np.concatenate([[y_value[:, 1]], [y_value[:, 0]]]).transpose()
                    y_predict = np.array(y_value > y_c, dtype=np.uint8)
                    
                    accuracy_score = np.divide(np.sum(np.multiply(ys, y_predict)), np.array(ys[:, 0]).size)
                    precision = np.divide(np.sum(np.multiply(ys[:, 0], y_predict[:, 0])), np.sum(y_predict[:, 0]))
                    recall = np.divide(np.sum(np.multiply(ys[:, 0], y_predict[:, 0])), np.sum(ys[:,0]))
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                    print("After %s training step(s), test precision = %g" % (global_step, precision))
                    print("After %s training step(s), test recall = %g" % (global_step, recall))
                    
                    coord.request_stop()
                    coord.join(threads)
                else:
                    print("No checkpoint found")
                    return time.sleep(TEST_INTERVAL_SECS)

def main():
    test()

if __name__ == '__main__':
    main()