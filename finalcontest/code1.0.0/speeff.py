#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:52:12 2019

@author: xudachengthu

Using "single PE' method the generate the answer
"""

import numpy as np
import h5py
import time
import standard
'''
fipt = "/Users/xudachengthu/Downloads/GHdataset/finalcontest_data/zincm-problem.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/submission/first-submission-spe-fin.h5"
'''
fipt = "/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/playground/first-submission-spe.h5"


def generate_eff():
    opd = [('EventID', '<i8'), ('ChannelID', '<i2'), ('PETime', 'f4'), ('Weight', 'f4')]
    model = generate_model(standard.single_pe_path)
    
    #model = np.add(model, 1)
    
    mtray = np.concatenate((np.zeros(400), model[0:50], np.zeros(350)))
    
    loperator = np.concatenate([mtray[400 - i : 800 - i] for i in range(400)]).reshape(400, 400)
    #loperator = loperator.astype(np.float64)
    #loperator = np.multiply(loperator, 100000)
    #print(np.linalg.matrix_rank(loperator))
    invl = np.linalg.inv(loperator)
    
    #a = np.matmul(loperator, invl)
    with h5py.File(fipt) as ipt, h5py.File(fopt, "w") as opt:
        ent = ipt['Waveform']
        l = len(ent)
        #l = 100
        print(l)
        dt = np.zeros(l*20, dtype=opd)
        start = 0
        end = 0
        count = 0
        for i in range(l):
            wf = ent[i]['Waveform']
            pf = np.matmul(wf[200:600], invl)
            
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
            opt.create_dataset('Answer', data=dt, compression='gzip')

def generate_model(spe_path):
    speFile = h5py.File(spe_path, 'r')
    spemean = np.mean(speFile['Sketchy']['speWf'], axis = 0)
    base_vol = np.mean(spemean[70:120])
    stdmodel = np.subtract(base_vol, spemean[20:120])
    stdmodel = np.multiply(np.around(np.divide(stdmodel, 0.05)), 0.05)
    
    #stdmodel = np.abs(np.where(stdmodel >= 0, stdmodel, 0))
    
    speFile.close()
    return stdmodel

def main():
    start_t = time.time()
    generate_eff()
    end_t = time.time()
    print(end_t - start_t)

if __name__ == '__main__':
    main()