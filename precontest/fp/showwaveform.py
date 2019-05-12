#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:28:48 2019

@author: xudachengthu

from /Users/xudachengthu/Desktop/GH/dcintro.pdf
show the data

/Users/xudachengthu/Downloads/GHdataset/first-problem.h5 --the file downloaded from crowdAI
"""

import h5py
filename = "/Users/xudachengthu/Downloads/GHdataset/first-problem.h5"
f = h5py.File(filename)
wfl = f['Waveform']
print(len(wfl))

ent = wfl[19656]
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

plt.title("Waveform")
plt.show()