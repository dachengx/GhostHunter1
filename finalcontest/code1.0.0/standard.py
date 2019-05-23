#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:35:57 2019

@author: xudachengthu

generate standard waveform and disperision of 'single incident pe - waveform'
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import h5py
#import os
import time

h5_path = '/Users/xudachengthu/Downloads/GHdataset/finalcontest_data/ztraining-0.h5'
single_pe_path = '/Users/xudachengthu/Downloads/GHdataset/sketchystore/single_pe.h5'
'''
h5_path = '/home/xudacheng/Downloads/GHdataset/finalcontest_data/ztraining-0.h5'
single_pe_path = '/home/xudacheng/Downloads/GHdataset/sketchystore/single_pe.h5'
'''
def generate_standard():
    opd_pe = [('EventID', 'u1'), ('ChannelID', 'u1'), ('Waveform', 'f', 1029), ('speWf', 'f', 120)]
    
    ztrfile = h5py.File(h5_path)
    
    ent = ztrfile['Waveform']
    answ = pd.read_hdf(h5_path, "GroundTruth")
    l = min(len(ent), 1000)
    print(l)
    ent = ent[0:l]
    answ = answ[0:20*l]
    dt = np.zeros(int(l/10), dtype = opd_pe)
    count = 0
    num = 0
    for i in range(l):
        eid = ent[i]['EventID']
        ch = ent[i]['ChannelID']
        pe = answ.query("EventID=={} & ChannelID=={}".format(eid, ch))
        #pe = answ[eid*300 : eid*500].query("EventID=={} & ChannelID=={}".format(eid, ch))
        pev = pe['PETime'].values
        unipe, c = np.unique(pev, return_counts=True)
        
        if np.size(unipe) == 1 and c[0] == 1:
            if unipe[0] < 21:
                print('opps!' + str(i))
            else:
                wf = ent[i]['Waveform']
                dt['speWf'][num] = wf[unipe[0] - 1 - 20 : unipe[0] - 1 + 100]
                dt['EventID'][num] = eid
                dt['ChannelID'][num] = ch
                dt['Waveform'][num] = wf
                # The 21th position is the pe incoming time
                num = num + 1
            
        count = count + 1
        if count == int(l / 100) + 1:
            print(int((i+1) / (l / 100)), end='% ', flush = True)
            count = 0
    
    print(num)
    dt = dt[np.where(dt['EventID'] > 0)]
    
    spemean = np.mean(dt['speWf'], axis = 0)
    plt.clf()
    plt.xlim(0,120)
    plt.ylim(930, 980)
    tr = list(range(0, 120))
    plt.plot(tr, spemean)
    plt.vlines([20], ymin=945, ymax=975)
    plt.xlabel('ns')
    plt.ylabel('mV')
    plt.savefig('spemean.eps')
    #plt.show()
    
    spemin = np.min(dt['speWf'],axis = 1)
    plt.clf()
    plt.hist(spemin, 50, density=1, histtype='bar', cumulative=True)
    plt.savefig('specumu.eps')
    #plt.show()
    
    plt.clf()
    plt.hist(spemin, 50, density=1, histtype='bar', cumulative=False)
    plt.savefig('speshow.eps')
    #plt.show()
    
    spp = h5py.File(single_pe_path, "w")
    spp.create_dataset('Sketchy', data=dt, compression='gzip')

def main():
    start_t = time.time()
    generate_standard()
    end_t = time.time()
    print(end_t - start_t)

if __name__ == '__main__':
    main()