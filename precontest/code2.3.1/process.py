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
import generate
import test

fipt = "/Users/xudachengthu/Downloads/GHdataset/first-problem.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/submission/first-submission-nn.h5"

fipt = "/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/playground/first-submission-nn.h5"

def restore_model(wf_test, wf_aver):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [1, 1, generate.Length_waveform, forward.NUM_CHANNELS])
        #y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        y = forward.forwardpro(x, False, None)
        
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        #saver = tf.train.Saver()
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                
                reshaped_xs = np.reshape(wf_test,(1, 1, 
                                             generate.Length_waveform, 
                                             forward.NUM_CHANNELS))
                    
                y_value = sess.run(y, feed_dict={x: reshaped_xs})
                #pe_num = np.around(np.polyval(test.REG, wf_aver))
                pe_num = np.around(np.polyval(test.REG, wf_aver)) * 0.55
                y_predict = np.zeros_like(y_value)
                
                order_y = np.argsort(y_value[0, :])[::-1]
                th_v = y_value[0, int(order_y[int(pe_num)])]
                y_predict = np.where(y_value > th_v, 1, 0)
                
                pf_value = np.where(y_predict == 1)[1]
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
        dt = np.zeros(l*20, dtype=opd)
        start = 0
        end = 0
        count = 0
        for i in range(l):
            wf = ent[i]['Waveform']
            af = np.where(wf[200:606] <= generate.THRES)
            
            if np.size(af) != 0:
                minit_v = af[0][0]
                tr = range(minit_v - 10 + 200, minit_v - 10 + generate.Length_waveform + 200)
                wf_test = wf[tr]
                
                wf_aver = np.mean(np.subtract(generate.PLATNUM, wf_test))/100
                pf = restore_model(wf_test.reshape([1, generate.Length_waveform]) * (1./100), wf_aver) + minit_v - 10 + 200
                
            if len(pf) == 0 or np.size(af) == 0:
                pf = np.array([300])
            
            end = start + len(pf)
            dt['PETime'][start:end] = pf
            dt['Weight'][start:end] = 1
            dt['EventID'][start:end] = ent[i]['EventID']
            dt['ChannelID'][start:end] = ent[i]['ChannelID']
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