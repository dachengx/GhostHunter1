#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:33:25 2019

@author: xudachengthu
"""

import h5py
filename1 = "/Users/xudachengthu/Downloads/GHdataset/first-problem.h5"
f = h5py.File(filename1)
wfl = f['Waveform']
print(len(wfl))

ent = wfl[535]
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
tr = range(200,606)
plt.plot(tr,w[tr])
plt.title('Waveform Zoomed')
plt.xlabel('ns')
plt.ylabel('mV')

plt.show()

print()