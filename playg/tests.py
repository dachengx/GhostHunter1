#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:10:37 2019

@author: xudachengthu

/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5 --the file downloaded from crowdAI
"""

import h5py
filename1 = "/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5"
filename2 = "/Users/xudachengthu/Downloads/GHdataset/playground/first-submission-nn.h5"
f = h5py.File(filename1)
wfl = f['Waveform']
print(len(wfl))

ent = wfl[4291]
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
tr = range(250,400)
plt.plot(tr,w[tr])
plt.title('Waveform Zoomed')
plt.xlabel('ns')
plt.ylabel('mV')

import pandas as pd
eid = ent['EventID']
ch = ent['ChannelID']
th = pd.read_hdf(filename2,"Answer")
th.head()

pe = th.query("EventID=={} & ChannelID=={}".format(ent["EventID"], ent["ChannelID"]))
pt = pe['PETime'].values
print(pt)

plt.vlines(pt, ymin=930, ymax=970)
plt.title("Waveform with Labels")
plt.show()

print()