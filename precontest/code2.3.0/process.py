#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 22:54:07 2019

@author: xudachengthu

partially from file:///Users/xudachengthu/Desktop/GH/threshold.html 
    and file:///Users/xudachengthu/Desktop/GH/process.html

batch dispose data

/Users/xudachengthu/Downloads/GHdataset/first-problem.h5 --the file downloaded from crowdAI
"""

import numpy as np
import tensorflow as tf
import h5py
import forward
import backward

fipt = "/Users/xudachengthu/Downloads/GHdataset/first-problem.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/submission/first-submission-thres.h5"

fipt = "/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/playground/first-submission-thres.h5"

def restore_model(wf):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [1, forward.INPUT_NODE])
        y = forward.forwardpro(x, None)
        '''
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        '''
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                
                pf_prob = sess.run(y, feed_dict={x: wf})
                pf_value = np.where(pf_prob > 0.5)[1] + 1 + 200
                return pf_value
            else:
                print("No checkpoint file found")
                return -1

def process_submit():
    opd = [('EventID', '<i8'), ('ChannelID', '<i2'), ('PETime', 'f4'), ('Weight', 'f4')]
    with h5py.File(fipt) as ipt, h5py.File(fopt, "w") as opt:
        ent = ipt['Waveform']
        l = len(ent)
        print(l)
        dt = np.zeros(l*500, dtype=opd)
        start = 0
        end = 0
        count = 0
        for i in range(l):
            wr = ent[i]
            pf = restore_model(np.array(wr['Waveform'], dtype=np.float32)[200:600].reshape([1, 400]) * (1./1000))
            if not len(pf):
                pf = np.array([300])
            end = start + len(pf)
            dt['PETime'][start:end] = pf
            dt['Weight'][start:end] = 1
            dt['EventID'][start:end] = wr['EventID']
            dt['ChannelID'][start:end] = wr['ChannelID']
            start = end
            count = count + 1
            if count == int(l / 100) + 1:
                print(int((i+1) / (l / 100)), end='% ')
                count = 0
        dt = dt[np.where(dt['EventID'] > 0)]
        opt.create_dataset('Answer', data=dt, compression='gzip')


def main():
    process_submit()

if __name__ == '__main__':
    main()