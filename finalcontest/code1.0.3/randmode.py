#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:32:38 2019

@author: xudachengthu

Find the random mode of the moise
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import pandas as pd
import h5py
#import os
import time

filename = "/Users/xudachengthu/Downloads/GHdataset/finalcontest_data/ztraining-0.h5"
'''
filename = "/home/xudacheng/Downloads/GHdataset/finalcontest_data/ztraining-0.h5"
'''
TEST_LEN = 50

def find_mode():
    f = h5py.File(filename)
    ent = f['Waveform']
    l = min(len(ent), 50)
    print(l)
    wf = ent[0 : int(l * 1.2)]['Waveform']
    dt = np.zeros((l, TEST_LEN))
    count = 0
    num = 0
    i = 0
    while num < l:
        sample = wf[i][-1 : -1 - TEST_LEN : -1]
        i = i + 1
        if np.size(np.where(sample < 960)) == 0:
            dt[num, 0:50] = sample
            num = num + 1
        
        count = count + 1
        if count == int(l / 100) + 1:
            print(int((i+1) / (l / 100)), end='% ', flush = True)
            count = 0
    f.close()
    print('At last, i = ' + str(i) + ' and count = ' + str(num))
    
    plt.clf()
    plt.hist(dt.flatten(), 20, density=1, histtype='bar', cumulative=True)
    plt.title('randommode, l=' + str(l))
    plt.savefig('randommode.eps')

def main():
    start_t = time.time()
    find_mode()
    end_t = time.time()
    print(end_t - start_t)

if __name__ == '__main__':
    main()