#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 07:53:13 2019

@author: xudachengthu

Using find peak method
"""

import numpy as np
import h5py
import time

fipt = "/Users/xudachengthu/Downloads/GHdataset/finalcontest_data/ztraining-0.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/submission/first-submission-spe-fin.h5"
'''
fipt = "/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/playground/first-submission-spe.h5"
'''
'''
fipt = "/home/xudacheng/Downloads/GHdataset/finalcontest_data/zincm-problem.h5"
fopt = "/home/xudacheng/Downloads/GHdataset/submission/first-submission-spe-fin.h5"

fipt = "/home/xudacheng/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/home/xudacheng/Downloads/GHdataset/playground/first-submission-spe.h5"
'''

THRES = 968

def findpeak_eff():
    opd = [('EventID', '<i8'), ('ChannelID', '<i2'), ('PETime', 'f4'), ('Weight', 'f4')]
    
    probh5 = h5py.File(fipt)
    ent = probh5['Waveform']
    l = min(len(ent), 1000)
    print(l)
    dt = np.zeros(l*20, dtype=opd)
    start = 0
    end = 0
    count = 0
    
    for i in range(l):
        wf = ent[i]['Waveform']
        
        wf = np.where(wf < THRES, wf , THRES)
        diff_wf = np.diff(wf)
        li = len(diff_wf)
        ptray = np.zeros([1, li - 1])
        for j in range(li - 1):
            if diff_wf[j] < 0 and diff_wf[j+1] > 0:
                ptray[0, j] = 1
        pf = np.where(ptray > 0)[1] - 15
        pf = pf[(pf > 0) & (pf <= 1029)]
        
        end = start + len(pf)
        dt['PETime'][start:end] = pf
        dt['Weight'][start:end] = 1
        dt['EventID'][start:end] = ent[i]['EventID']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
        
        count = count + 1
        if count == int(l / 100) + 1:
            print(int((i+1) / (l / 100)), end = '% ', flush=True)
            count = 0
    probh5.close()
    
    dt = dt[np.where(dt['EventID'] > 0)]
    opt = h5py.File(fopt, "w")
    opt.create_dataset('Answer', data = dt, compression='gzip')

def main():
    start_t = time.time()
    findpeak_eff()
    end_t = time.time()
    print(end_t - start_t)

if __name__ == '__main__':
    main()