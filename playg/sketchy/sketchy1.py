#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:57:25 2019

@author: xudachengthu

find out the Simple Features of playground-data.h5 -- MPES
"""

import numpy as np, pandas as pd, h5py

filename1 = "/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5"
filename2 = "/Users/xudachengthu/Downloads/GHdataset/playground/submission-example.h5"
playd = h5py.File(filename1)
ent = playd['Waveform']
answ = pd.read_hdf(filename2, "Answer")

l = min(len(ent),100000)
print(l)
se = np.linspace(1, 10, 10, dtype='int32')
countall = np.asarray((se, np.zeros_like(se, dtype='int32'))).T
for i in range(l):
    eid = ent[i]['EventID']
    ch = ent[i]['ChannelID']
    pe = answ.query("EventID=={} & ChannelID=={}".format(eid, ch))
    _, c = np.unique(pe['PETime'].values, return_counts=True)
    unique, counts = np.unique(c,return_counts=True)
    for j in range(len(unique)):
        countall[unique[j]-1,1] = countall[unique[j]-1,1] + counts[j]
    #countall = np.concatenate((countall,np.asarray((unique,counts)).T))

#unique, counts = np.unique(countall[1:,1], return_counts=True)
#res = np.asarray((unique,counts)).T
#print(res.tolist())
print(countall)

playd.close()