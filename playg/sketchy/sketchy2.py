#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:49:50 2019

@author: xudachengthu

find out the Simple Features of playground-data.h5 -- AEWL

find out the Simple Features of playground-data.h5 -- AEWH & AEWT

find out the Simple Features of playground-data.h5 -- MVR
"""

import numpy as np, h5py

filename1 = "/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5"

playd = h5py.File(filename1)
ent = playd['Waveform']

l = len(ent)
print(l)
minmaxvall = np.zeros([l,2], dtype = 'int16')
minmaxtall = np.zeros([l,3], dtype = 'int16')
thres = 968
print(thres)
for i in range(l):
    wf = ent[i]['Waveform']
    miniv = np.min(wf)
    maxiv = np.max(wf)
    minmaxvall[i,0] = miniv
    minmaxvall[i,1] = maxiv
    
    # thres can be changed
    w = np.array(wf < thres, dtype=np.int8)
    d = np.diff(w)
    # latter one subtract former one
    minit = np.where(d >= 1)[0][0]
    maxit = np.where(d <= -1)[0][-1]
    lent = maxit - minit
    minmaxtall[i,0] = minit
    minmaxtall[i,1] = maxit
    minmaxtall[i,2] = lent

minv = np.min(minmaxvall[:,0])
maxv = np.max(minmaxvall[:,1])

mint = np.min(minmaxtall[:,0])
imint = np.argmin(minmaxtall[:,0])
maxt = np.max(minmaxtall[:,1])
imaxt = np.argmax(minmaxtall[:,1])
minlent = np.min(minmaxtall[:,2])
maxlent = np.max(minmaxtall[:,2])
meanlent = np.mean(minmaxtall[:,2])

print(minv,maxv)
print(mint,imint,maxt,imaxt,minlent,maxlent,meanlent)

playd.close()