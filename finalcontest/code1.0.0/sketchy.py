#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 23:16:34 2019

@author: duncan

find all the waveform which only gives one pe output
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os

h5_path = '/home/duncan/Downloads/GHdataset/ftraining-0.h5'

def show_onepe():
    h5file = h5py.File(h5_path)
    ent = h5file['Waveform']
    answ = pd.read_hdf(h5_path, "GroundTruth")
    l = min(len(ent),500000)
    print(l)
    ent = ent[0:l]
    answ = answ[0:20*l]
    count = 0
    for i in range(l):
        eid = ent[i]['EventID']
        ch = ent[i]['ChannelID']
        pe = answ.query("EventID=={} & ChannelID=={}".format(eid, ch))
        pev = pe['PETime'].values
        unipe, c = np.unique(pev, return_counts=True)
        wf = ent[i]['Waveform']
        
        if np.size(unipe) == 1 and c[0] == 1:
            plt.clf()
            plt.xlim(250,500)
            plt.ylim(930, 980)
            tr = list(range(200, 500 + 1))
            plt.plot(tr, wf[tr])
            plt.vlines(unipe[0], ymin=945, ymax=975)
            plt.title('Waveform Zoomed in 250-500, i = ' + str(i) + 'ch = ' + str(ch))
            plt.xlabel('ns')
            plt.ylabel('mV')
            plt.savefig('./skshow/channel' + str(ch) + '/pe1ent' + str(i) + 'ch' + str(ch) + '.eps')
            #plt.show()
        count = count + 1
        if count == int(l / 100) + 1:
            print(int((i+1) / (l / 100)), end='% ', flush = True)
            count = 0
        
        h5file.close()

def gene_onepe():
        for i in range(0,30):
            isExists = os.path.exists('./skshow/channel' + str(i))
            if not isExists:
                os.makedirs('./skshow/channel' + str(i))
                print('The directory ./skshow/channel' + str(i) + ' has created successfully')
            else:
                print('The directory ./skshow/channel' + str(i) + ' has already been created')
        show_onepe()

def main():
    gene_onepe()

if __name__ == '__main__':
    main()
