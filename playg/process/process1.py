#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:06:47 2019

@author: xudachengthu

from file:///Users/xudachengthu/Desktop/GH/threshold.html 
    and file:///Users/xudachengthu/Desktop/GH/process.html

batch dispose data

/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5 --the file downloaded from crowdAI
"""

import numpy as np, h5py
fipt = "/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5"
fopt = "first-submission-thres.h5"

opd = [('EventID', '<i8'), ('ChannelID', '<i2'),
       ('PETime', 'f4'), ('Weight', 'f4')]

def mmp(wr):
    w01i = np.array(wr['Waveform']<962, dtype=np.int8)
    d01i = np.diff(w01i)
    pf = np.where(d01i>=1)[0]
    
    if not len(pf):
        pf = np.array([300])
    rst = np.zeros(len(pf), dtype=opd)
    rst['PETime'] = pf - 6
    rst['Weight'] = 1
    rst['EventID'] = wr['EventID']
    rst['ChannelID'] = wr['ChannelID']
    return rst

with h5py.File(fipt) as ipt, h5py.File(fopt, "w") as opt:
    dt = np.concatenate([mmp(wr) for wr in ipt['Waveform']])
    opt.create_dataset('Answer', data=dt, compression='gzip')