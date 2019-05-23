#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 19:47:03 2019

@author: xudachengthu

Using "single PE' method the generate the answer
"""

import numpy as np
import tensorflow as tf
import h5py
import time
import standard
'''
fipt = "/Users/xudachengthu/Downloads/GHdataset/finalcontest_data/zincm-problem.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/submission/first-submission-spe-fin.h5"
'''
fipt = "/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/playground/first-submission-spe.h5"

'''
fipt = "/home/xudacheng/Downloads/GHdataset/finalcontest_data/zincm-problem.h5"
fopt = "/home/xudacheng/Downloads/GHdataset/submission/first-submission-spe-fin.h5"
'''
fipt = "/home/xudacheng/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/home/xudacheng/Downloads/GHdataset/playground/first-submission-spe.h5"

LEARNING_RATE = 0.01
STEPS = 10000
REG_RAW = [84.28899 * 2 / 3, 0]
THRES = 968
PLATNUM = 976


def generate_eff():
    opd = [('EventID', '<i8'), ('ChannelID', '<i2'), ('PETime', 'f4'), ('Weight', 'f4')]
    model = generate_model(standard.single_pe_path)
    
    mtray = np.concatenate((np.zeros(400), model[0:50], np.zeros(350)))
    
    loperator = np.concatenate([mtray[400 - i : 800 - i] for i in range(400)]).reshape(400, 400)
    #invl = np.linalg.inv(loperator)
    with h5py.File(fipt) as ipt, h5py.File(fopt, "w") as opt:
        ent = ipt['Waveform']
        l = len(ent)
        l = 100
        print(l)
        dt = np.zeros(l*20, dtype=opd)
        start = 0
        end = 0
        count = 0
        for i in range(l):
            wf = ent[i]['Waveform']
            wf_input = np.subtract(np.mean(wf[900:1000]), wf[200 : 600]).reshape(1,400)
            #pf = np.matmul(wf[200:600], invl)
            
            with tf.Graph().as_default():
                y_ = tf.placeholder(tf.float32, shape=(1, 400))
                LO = tf.placeholder(tf.float32, shape=(400, 400))
                w = tf.Variable(tf.random_uniform([1, 400], minval=0.0, maxval=1.0))
                wsq = tf.multiply(w, w)
                y = tf.matmul(wsq, LO)

                loss = tf.reduce_mean(tf.square(y_ - y))
                
                train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
                
                with tf.Session() as sess:
                    init_op = tf.global_variables_initializer()
                    sess.run(init_op)
                    
                    for j in range(STEPS):
                        _, loss_value, w_val, wsq_val = sess.run([train_step, loss, w, wsq], feed_dict = {LO : loperator, y_ : wf_input})
                        
                        if j % 100 == 0:
                            print("After %d training step(s), loss on training batch is %g." % (j, loss_value))
                
                af = np.where(wf[200:606] <= THRES)
                
                if np.size(af) != 0:
                    minit_v = af[0][0]
                    tr = range(minit_v - 10 + 200, minit_v - 10 + 206 + 200)
                    wf_test = wf[tr]
                
                wf_aver = np.mean(np.subtract(PLATNUM, wf_test)) * (1./100)
                pe_num = int(np.around(np.polyval(np.array(REG_RAW), wf_aver)))
                y_predict = np.zeros_like(wsq_val)
                order_y = np.argsort(wsq_val[0, :])[::-1]
                th_v = wsq_val[0, int(order_y[pe_num])]
                y_predict = np.where(wsq_val > th_v, 1, 0)
                
                pf = np.where(y_predict == 1)[1] + 200
            
            end = start + len(pf)
            dt['PETime'][start:end] = pf
            dt['Weight'][start:end] = 1
            dt['EventID'][start:end] = ent[i]['EventID']
            dt['ChannelID'][start:end] = ent[i]['ChannelID']
            start = end
            count = count + 1
            if count == int(l / 100) + 1:
                print(int((i+1) / (l / 100)), end='% ', flush=True)
                count = 0
                
        dt = dt[np.where(dt['EventID'] > 0)]
        opt.create_dataset('Answer', data = dt, compression='gzip')

def generate_model(spe_path):
    speFile = h5py.File(spe_path, 'r')
    spemean = np.mean(speFile['Sketchy']['speWf'], axis = 0)
    base_vol = np.mean(spemean[70:120])
    stdmodel = np.subtract(base_vol, spemean[20:120])
    stdmodel = np.multiply(np.around(np.divide(stdmodel, 0.05)), 0.05)
    
    stdmodel = np.abs(np.where(stdmodel >= 0, stdmodel, 0))
    
    speFile.close()
    return stdmodel

def main():
    start_t = time.time()
    generate_eff()
    end_t = time.time()
    print(end_t - start_t)

if __name__ == '__main__':
    main()