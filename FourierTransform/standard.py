#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:51:47 2019

@author: xudachengthu

Generate standard response model of single pe's waveform
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import time

def generate_standard(h5_path, single_pe_path):
    npdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint8), ('Waveform', np.uint16, 1029), ('speWf', np.uint16, 120)]) # set datatype
    
    ztrfile = h5py.File(h5_path) # read h5 file
    
    ent = ztrfile['Waveform'] # read waveform only
    answ = pd.read_hdf(h5_path, "GroundTruth") # read h5 file answer
    l = min(len(ent), 1000) # limit l to below 5, l is the amount of event
    ent = ent[0 : l]
    answ = answ[0 : 20*l] # assume 1 waveform has less than 20 answers
    dt = np.zeros(int(l/10), dtype = npdt) # assume 10 Events has less than 1 single pe event among them
    count = 0
    num = 0
    
    for i in range(l):
        eid = ent[i]['EventID']
        ch = ent[i]['ChannelID'] # in some Event, the amount of Channel < 30
        pe = answ.query("EventID=={} & ChannelID=={}".format(eid, ch))
        pev = pe['PETime'].values # fetch corresponding PETime to the specific EventID & ChannelID
        unipe, c = np.unique(pev, return_counts=True)
        
        if np.size(unipe) == 1 and c[0] == 1: # if single pe
            if unipe[0] < 21 or unipe[0] > 930:
                print('opps! ' + eid) # print Event when the single pe is too early or too late
            else:
                wf = ent[i]['Waveform'] # temporarily record waveform
                dt['speWf'][num] = wf[unipe[0] - 1 - 20 : unipe[0] - 1 + 100] # only record 120 position, and the time of spe is the 21th
                dt['EventID'][num] = eid
                dt['ChannelID'][num] = ch
                dt['Waveform'][num] = wf
                # The 21th position is the spe incoming time
                num = num + 1 # preparing for next record
            
        count = count + 1
        if count == int(l / 100) + 1:
            print(int((i+1) / (l / 100)), end='% ', flush = True)
            count = 0 # show the progress
    
    

def main():
    start_t = time.time()
    h5_path = '/home/xudacheng/Downloads/GHdataset/finalcontest_data/ztraining-0.h5'
    single_pe_path = '/home/xudacheng/Downloads/GHdataset/sketchystore/single_pe.h5'
    generate_standard(h5_path, single_pe_path) # generate response model
    end_t = time.time()
    print(end_t - start_t)

if __name__ == '__main__':
    main()