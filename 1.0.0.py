#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:59:12 2019

@author: gutianren
"""

"""
A basic logistic-regression model with Y = sigmoid(WX + b) using Tensorflow
W is the convolution core to evaluate Wave[t - delta, t + delta)
However it turns out to be a huge mistake as I do not consider the Time offset
And it seems that tensorflow cannot train an integer time offset ?
Perhaps we should to change the nn model or choose a more reasonable convolution core ?
Another problem is that it runs too slow so I can only use a little of the 700MB Training set
Rewrite Train() and getX() / getY() if needed
Hint : I have some Timer(mostly cnt) in the program, just neglect some display bugs
"""

import numpy as np, h5py
import tensorflow as tf

# Some basic constants

fipt_train = "ftraining-0.h5"
fipt = "first-problem.h5"
fopt = "first-submission-thres.h5"
opd = [('EventID','<i8'),('ChannelID','<i2'),('PETTime','f4'),('Weight','f4')]
MAXN = 1000  # Training Sets
MAXM = 100000  # Predicting Sets
CHANNEL_MAXN = 30
WAVELEN = 1029
delta = 10  # Convolution Core
LEARNING_CNT = 1000
LEARNING_RATE = 0.01

def normalization(X):
    Temp = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
    return np.abs(Temp)

def split(wrw, delta, count = False):
    if count:
        global cnt
        cnt += 1
        if cnt % (MAXN // 1000) == 0:
            print('\r',round(cnt / (MAXN // 100),2),'%', end = '')
    Temp = []
    n = len(wrw)
    temp = np.arange(- delta, n - delta)

    for i in range(2 * delta):
        if i >= delta:
            temp[n - 1 - i + delta] -= n
        Temp = np.append(Temp, wrw[temp])
        temp += 1
    
    Temp = Temp.reshape(2 * delta, n)
 #   if cnt == 1:    
 #       print(Temp.shape)
    return Temp.T

def getX(ipt, delta):
    return np.concatenate([split(wr['Waveform'], delta, count = True) for wr in ipt['Waveform'][0:MAXN]])

def getY(ipt, delta):
    wfl = ipt['Waveform']
    Temp = np.zeros(MAXN * WAVELEN)
    gt = ipt['GroundTruth']
    j = 0
    for i in range(MAXN):
        while gt[j]['EventID'] <= wfl[i]['EventID'] and gt[j]['ChannelID'] <= wfl[i]['ChannelID']:
            if gt[j]['EventID'] == wfl[i]['EventID'] and gt[j]['ChannelID'] == wfl[i]['ChannelID']:
                Temp[i * WAVELEN + gt[j]['PETime']] = 1
            j += 1
            if i == MAXN - 1 and gt[j]['EventID'] > wfl[i]['EventID']:
                break
        if i % (MAXN // 1000) == 0:
            print('\r',round(i / (MAXN // 100),2),'%', end = '')
        
    return np.reshape(Temp,[-1,1])    

def Train(X_train, Y_train):
    
    print("Training...")

#    n_samples = len(X_train)
    n_feature = len(X_train[0])
    
    X = tf.placeholder(tf.float32, name = 'X')
    Y = tf.placeholder(tf.float32, name = 'Y')
    
    W = tf.Variable(tf.zeros([n_feature, 1]))
    b = tf.Variable([-1.0])
    
    db = tf.matmul(X, tf.reshape(W, [-1,1])) + b

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = db)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for step in range(LEARNING_CNT):
        sess.run([optimizer,loss], feed_dict = {X:X_train, Y:Y_train})
        print('\r',step * 100 / LEARNING_CNT,'%',end = '')
    print('\r')

    W_value, b_value = sess.run([W,b])
    return W_value, b_value

def npsigmoid(X):
    return 1.0 / (1 + np.exp(-X))

def predict(W, b, wr, delta):
    
    X = normalization(split(wr['Waveform'],delta))
    Y_pred = np.concatenate(npsigmoid(np.matmul(X, np.reshape(W,[-1,1])) + b))
    pf = np.where(Y_pred >= .5)[0]
    global cnt
    cnt += 1
    if cnt % (MAXM // 1000) == 0:
        print('\r',round(cnt / (MAXM // 100),2),'%', end = '')

    if not len(pf):
        pf = np.array([300])
    
    rst = np.zeros(len(pf), dtype = opd)
    rst['PETTime'] = pf
    rst['Weight'] = Y_pred[pf]
    rst['EventID'] = wr['EventID']
    rst['ChannelID'] = wr['ChannelID']
    
    return rst

with h5py.File(fipt_train) as ipt:
    print('Pre-Treatment...')
    cnt = 0
    print('Getting X...')
    X_train = normalization(getX(ipt,delta))
    print()
    print('Getting Y...')
    Y_train = getY(ipt,delta)
    print()
#    print('X_train = ', len(X_train))
#    print('Y_train = ', len(Y_train))
    W_value, b_value = Train(X_train, Y_train)
    print('Trainning Ended...')

with h5py.File(fipt) as ipt, h5py.File(fopt,"w") as opt:
#    dt = np.concatenate([mmp(wr) for wr in ipt['Waveform']])
#    opt.create_dataset('Answer', data = dt, compression = 'gzip')
    cnt = 0
    print('Beginning Predicting...')
    dt = np.concatenate([predict(W_value, b_value, wr, delta) for wr in ipt['Waveform'][0:MAXM]])
    opt.create_dataset('Answer', data = dt, compression = 'gzip')
