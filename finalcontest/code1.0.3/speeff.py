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
import matplotlib.pyplot as plt

fipt = "/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/playground/first-submission-spe.h5"

'''
fipt = "/home/xudacheng/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/home/xudacheng/Downloads/GHdataset/playground/first-submission-spe.h5"
'''
LEARNING_RATE = 0.005
STEPS = 5000
Length_pe = 200
THRES = 968
PLATNUM = 976
#NORMAL_P = 30
BATCH_SIZE = 100
GRAIN = 0.05

def generate_eff():
    opd = [('EventID', '<i8'), ('ChannelID', '<i2'), ('PETime', 'f4'), ('Weight', 'f4')]
    model = generate_model(standard.single_pe_path)
    
    mtray = np.concatenate((np.zeros(Length_pe), model[0 : 50], np.zeros(Length_pe)))
    
    loperator = np.concatenate([mtray[Length_pe - i : 2 * Length_pe + 50 - i] for i in range(Length_pe)]).reshape(Length_pe, Length_pe + 50)
    #invl = np.linalg.inv(loperator)
    with h5py.File(fipt) as ipt, h5py.File(fopt, "w") as opt:
        ent = ipt['Waveform']
        l = len(ent)
        #l = 70
        print(l)
        dt = np.zeros(l*200, dtype=opd)
        start = 0
        end = 0
        count = 0
        
        with tf.Graph().as_default():
            y_ = tf.placeholder(tf.float32, shape=(BATCH_SIZE, Length_pe + 50))
            LO = tf.placeholder(tf.float32, shape=(Length_pe, Length_pe + 50))
            w = tf.Variable(tf.random_uniform([BATCH_SIZE, Length_pe], minval=0.0, maxval=1.0))
            #w = tf.Variable(tf.zeros([BATCH_SIZE, Length_pe]))
            
            wsq = tf.multiply(w, w)
            #wsq = w
            
            y = tf.matmul(wsq, LO)
            loss = tf.reduce_mean(tf.square(y_ - y))
            train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
            
            chunk = int(l/BATCH_SIZE) + 1
            for i in range(chunk):
                
                if i <= int(l/BATCH_SIZE) - 1:
                    size_out = BATCH_SIZE
                    wf = ent[i * BATCH_SIZE : (i+1) * BATCH_SIZE]['Waveform']
                    eid_out = ent[i * BATCH_SIZE : (i+1) * BATCH_SIZE]['EventID']
                    chid_out = ent[i * BATCH_SIZE : (i+1) * BATCH_SIZE]['ChannelID']
                else:
                    size_out = l % BATCH_SIZE
                    wf = ent[i * BATCH_SIZE : l]['Waveform']
                    wf = np.concatenate([wf, np.zeros((BATCH_SIZE - size_out, 1029))])
                    eid_out = ent[i * BATCH_SIZE : l]['EventID']
                    chid_out = ent[i * BATCH_SIZE : l]['ChannelID']
                
                wf_input = np.zeros((BATCH_SIZE, Length_pe + 50))
                minit_v_tray = np.zeros((BATCH_SIZE, 1))
                for j in range(BATCH_SIZE):
                    
                    af = np.where(wf[j, 200:600] <= THRES)
                    if np.size(af) != 0:
                        minit_v_tray[j, 0] = af[0][0]
                    else:
                        minit_v_tray[j, 0] = 210
                    tr = range(int(minit_v_tray[j, 0]) - 10 + 200, int(minit_v_tray[j, 0]) - 10 + Length_pe + 50 + 200)
                    wf_test = wf[j, :][tr]
                    wf_input[j, :] = np.subtract(np.mean(wf[j, 900:1000]), wf_test).reshape(1, Length_pe + 50)
                    
                with tf.Session() as sess:
                    init_op = tf.global_variables_initializer()
                    sess.run(init_op)
                    
                    for k in range(STEPS):
                        _, loss_value, w_val, wsq_val, L = sess.run([train_step, loss, w, wsq, LO], feed_dict = {LO : loperator, y_ : wf_input})
                        '''
                        if k % 100 == 0:
                            print("After %d training step(s), loss on training batch is %g." % (k, loss_value))
                '''
                for j in range(size_out):
                    pf = np.multiply(np.around(np.divide(wsq_val[j, :], GRAIN)), GRAIN)
                    '''
                    plt.clf()
                    plt.plot(np.matmul(pf, loperator))
                    plt.title("Waveform")
                    plt.xlabel('ns')
                    plt.ylabel('mV')
                    plt.show()
                    '''
                    
                    lenpf = np.size(np.where(pf > 0))
                    if lenpf == 0:
                        pf[0] = 1
                    lenpf = np.size(np.where(pf > 0))
                    pet = np.where(pf > 0) + minit_v_tray[j, 0] - 10 + 200
                    pwe = pf[pf > 0]
                    end = start + lenpf
                    dt['PETime'][start:end] = pet
                    dt['Weight'][start:end] = pwe
                    dt['EventID'][start:end] = eid_out[j]
                    dt['ChannelID'][start:end] = chid_out[j]
                    start = end
                    
                count = count + 1
                if count == int(chunk / 100) + 1:
                    print(int((i+1) / (chunk / 100)), end = '% ', flush=True)
                    count = 0
                
        dt = dt[np.where(dt['Weight'] > 0)]
        opt.create_dataset('Answer', data = dt, compression='gzip')
        print(fopt, end = ' ', flush=True)

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