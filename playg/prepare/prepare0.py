#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 00:34:01 2019

@author: xudachengthu

from /Users/xudachengthu/Desktop/GH/dcintro.pdf
show the data

/Users/xudachengthu/Downloads/GHdataset/ftraining-0.h5 --the file downloaded from crowdAI
"""

import h5py
filename = "/Users/xudachengthu/Downloads/GHdataset/finalcontest_data/ztraining-0.h5"
f = h5py.File(filename)
wfl = f['Waveform']
print(len(wfl))

ent = wfl[291379]
f.close()
w = ent['Waveform']
print(len(w))

from matplotlib import pyplot as plt
plt.clf()
plt.plot(w)
plt.title("Waveform")
plt.xlabel('ns')
plt.ylabel('mV')
plt.show()

plt.clf()
plt.plot(w)
plt.title("Waveform")
plt.xlabel('ns')
plt.ylabel('mV')
plt.show()

plt.clf()
tr = range(250,400)
plt.plot(tr, w[tr])
plt.title('Waveform Zoomed')
plt.xlabel('ns')
plt.ylabel('mV')

import pandas as pd
eid = ent['EventID']
ch = ent['ChannelID']
th = pd.read_hdf(filename,"GroundTruth")
th.head()

pe = th.query("EventID=={} & ChannelID=={}".format(ent["EventID"], ent["ChannelID"]))
pt = pe['PETime'].values
print(pt)

plt.vlines(pt, ymin=930, ymax=970)
plt.title("Waveform with Labels")

plt.show()
print()