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
import pandas as pd
import h5py
#import os
import time

filename = "/Users/xudachengthu/Downloads/GHdataset/finalcontest_data/ztraining-0.h5"

filename = "/home/xudacheng/Downloads/GHdataset/finalcontest_data/ztraining-0.h5"

def findmode():
    f = h5py.File(filename)
    ent = f['Waveform']
    l = min(len(ent), 500000)
    print(l)
    dt = np.zeros((l, 50))
    num = 0
    while num < l:
        

def main():
    start_t = time.time()
    find_mode()
    end_t = time.time()
    print(end_t - start_t)

if __name__ == '__main__':
    main()