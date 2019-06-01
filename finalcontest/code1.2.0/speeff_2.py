#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 21:53:05 2019

@author: xudachengthu

Test the performanca of different parameters -- 2
"""

import numpy as np
import tensorflow as tf
import h5py
import time
import standard
import matplotlib.pyplot as plt
import threading

fipt = "/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5"
fopt_prefix = "/Users/xudachengthu/Downloads/GHdataset/playground/"

'''
fipt = "/home/xudacheng/Downloads/GHdataset/playground/playground-data.h5"
fopt_prefix = "/home/xudacheng/Downloads/GHdataset/playground/"
'''
LEARNING_RATE = 0.005
#STEPS = 5000
STEPS = [10000]
Length_pe = 200
THRES = 968
BATCH_SIZE = 100
#GRAIN = 0.05

KNIFE = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
#KNIFE = [0.01]

THREADS = 5

class speeffThread(threading.Thread):
    def __int__(self, fopt, threadID, knife, steps, start, end):
        threading.Thread.__init__(self)
        self.fopt = fopt
        self.threadID = threadID
        self.knife = knife
        self.steps = steps
        self.start = start
        self.end = end
    def run(self):
        print ("Start Thread " + self.threadID)
        generate_eff_test(self.fopt, self.threadID, self.knife, self.steps, self.start, self.end)
        print ("End Thread " + self.threadID)

def generate_eff_test(fopt, fID, knife, steps, start, end):
    fopt_t = fopt + '-' + str(fID) + '.h5'
    
    opd = [('EventID', '<i8'), ('ChannelID', '<i2'), ('PETime', 'f4'), ('Weight', 'f4')]
    model = generate_model(standard.single_pe_path)
    
    mtray = np.concatenate((np.zeros(Length_pe), model[0 : 50], np.zeros(Length_pe)))
    
    loperator = np.concatenate([mtray[Length_pe - i : 2 * Length_pe + 50 - i] for i in range(Length_pe)]).reshape(Length_pe, Length_pe + 50)
    #invl = np.linalg.inv(loperator)
    with h5py.File(fipt, 'r') as ipt, h5py.File(fopt_t, "w") as opt_t:
        ent = ipt['Waveform']
        ent = ent[start : end]
        l = len(ent)
        #l = 70
        print(l)
        dt = np.zeros(l*Length_pe, dtype=opd)
        start = 0
        end = 0
        #ount = 0
        
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
                    
                    for k in range(steps):
                        _, loss_value, w_val, wsq_val, L = sess.run([train_step, loss, w, wsq, LO], feed_dict = {LO : loperator, y_ : wf_input})
                        '''
                        if k % 100 == 0:
                            print("After %d training step(s), loss on training batch is %g." % (k, loss_value))
                    
                print("After %d training steps, loss on training batch is %g." % (steps, loss_value))
                '''
                for j in range(size_out):
                    '''
                    pf = np.multiply(np.around(np.divide(wsq_val[j, :], grain)), grain)
                    '''
                    pf = wsq_val[j, :]
                    pf = np.where(pf > knife, pf, 0)
                    
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
                    '''
                count = count + 1
                if count == int(chunk / 100) + 1:
                    print(int((i+1) / (chunk / 100)), end = '% ', flush=True)
                    count = 0
                '''
        dt = dt[np.where(dt['Weight'] > 0)]
        
        opt_t.create_dataset('Answer', data = dt, compression='gzip')
        print(fopt_t, end = ' ', flush=True)

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
    F = h5py.File(fipt, 'r')
    ent = F['Waveform']
    lg = len(ent)
    F.close()
    loaf = int(lg/THREADS)
    for i in range(len(KNIFE)):
        for j in range(len(STEPS)):
            fopt = fopt_prefix + 'k' + str(i) + '-' + str(j)
            start_t = time.time()
            
            start_p = np.zeros(THREADS)
            end_p = np.zeros(THREADS)
            for k in range(THREADS):
                start_p[k] = k * loaf
                end_p[k] = start_p[k] + loaf
            end_p[THREADS - 1] = lg
            
            thread1 = speeffThread(fopt, 0, KNIFE[i], STEPS[j], start_p[0], end_p[0])
            thread2 = speeffThread(fopt, 1, KNIFE[i], STEPS[j], start_p[1], end_p[1])
            thread3 = speeffThread(fopt, 2, KNIFE[i], STEPS[j], start_p[2], end_p[2])
            thread4 = speeffThread(fopt, 3, KNIFE[i], STEPS[j], start_p[3], end_p[3])
            thread5 = speeffThread(fopt, 4, KNIFE[i], STEPS[j], start_p[4], end_p[4])
            
            thread1.start()
            thread2.start()
            thread3.start()
            thread4.start()
            thread5.start()
            
            thread1.join()
            thread2.join()
            thread3.join()
            thread4.join()
            thread5.join()
            
            bipt0 = h5py.File(fopt + '-0.h5')['Answer']
            bipt1 = h5py.File(fopt + '-1.h5')['Answer']
            bipt2 = h5py.File(fopt + '-2.h5')['Answer']
            bipt3 = h5py.File(fopt + '-3.h5')['Answer']
            bipt4 = h5py.File(fopt + '-4.h5')['Answer']
            
            dt = np.concatenate([bipt0, bipt1, bipt2, bipt3, bipt4])
            
            opt = h5py.File(fopt + '.h5', "w")
            opt.create_dataset('Answer', data = dt, compression='gzip')
            
            end_t = time.time()
            print('Time for ' + str(KNIFE[i]) + ' ' + str(STEPS[j]) + ' is ' + str(end_t - start_t))

if __name__ == '__main__':
    main()