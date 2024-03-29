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
import forward
import backward
import generate
TEST_INTERVAL_SECS = 10
TEST_NUM = 500

def test():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        y = forward.forwardpro(x, None)
        '''
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        '''
        saver = tf.train.Saver()
        correct_prediction = tf.equal(y_, tf.add(tf.div(tf.sign(tf.subtract(y,0.5)),2),0.5))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
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
                    
                    accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                    
                    coord.request_stop()
                    coord.join(threads)
                else:
                    print("No checkpoint found")
                    return time.sleep(TEST_INTERVAL_SECS)

def main():
    test()

if __name__ == '__main__':
    main()