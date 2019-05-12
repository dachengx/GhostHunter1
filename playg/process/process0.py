#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:41:46 2019

@author: xudachengthu

from file:///Users/xudachengthu/Desktop/GH/threshold.html
data dispose prototype

/Users/xudachengthu/Downloads/GHdataset/ftraining-0.h5 --the file downloaded from crowdAI
"""

filename = '/Users/xudachengthu/Downloads/GHdataset/ftraining-0.h5'

import h5py
f = h5py.File(filename)
wfl = f['Waveform']
len(wfl)
ent = wfl[233]
f.close()
w=ent['Waveform']

import pandas as pd
eid=ent['EventID']
ch=ent['ChannelID']
th = pd.read_hdf(filename,'GroundTruth')
pe = th.query("EventID=={} & ChannelID=={}".format(ent["EventID"], ent["ChannelID"]))
pt = pe['PETime'].values

from matplotlib import pylab as plt
plt.clf();
tr = range(250, 400)
plt.plot(tr, w[tr])
plt.xlabel('ns'); plt.ylabel('mV')
plt.vlines(pt, ymin=930, ymax=970)
plt.title("Waveform with Labels")
plt.show()

w01 = w<962
plt.plot(tr, w01[tr])
plt.title("01 Series after Thresholding")
plt.ylabel("Lower than 962")
plt.show()

import numpy as np
plt.plot(tr, np.diff(w01)[tr])
plt.show()

w01i = np.array(w01, dtype=np.int8)
d01i = np.diff(w01i)
plt.plot(tr, d01i[tr])
plt.show()

pf = np.where(d01i>=1)[0]
plt.plot(tr, w[tr])
plt.xlabel('ns'); plt.ylabel('mV')
plt.vlines(pf, ymin=930, ymax=970)
plt.title("PE Search by Thresholds")
plt.show()

print(pt)
print(pf)

pf = np.where(d01i>=1)[0] - 5
plt.plot(tr, w[tr])
plt.xlabel('ns'); plt.ylabel('mV')
plt.vlines(pf, ymin=930, ymax=970)
plt.title("PE Search by Thresholds")

plt.show()