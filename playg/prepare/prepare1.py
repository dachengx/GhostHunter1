#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:21:53 2019

@author: xudachengthu

from https://ghost-hunter.net9.org/2019/02/01/fs.html
show the data

/Users/xudachengthu/Downloads/GHdataset/ftraining-0.h5 --the file downloaded from crowdAI
"""

import tables
#import matplotlib
import matplotlib.pyplot as plt

# Read hdf5 file
filename = "/Users/xudachengthu/Downloads/GHdataset/ftraining-0.h5"
h5file = tables.open_file(filename, "r")

WaveformTable = h5file.root.Waveform
entry = 0
EventId = WaveformTable[entry]['EventID']
ChannelId = WaveformTable[entry]['ChannelID']
Waveform = WaveformTable[entry]['Waveform']
minpoint = min(Waveform)
maxpoint = max(Waveform)

GroundTruthTable = h5file.root.GroundTruth
PETime = [x['PETime'] for x in GroundTruthTable.iterrows() if x['EventID'] == EventId and x['ChannelID']==ChannelId]
print(PETime)

plt.plot(Waveform)
plt.xlabel('Time [ns]')
plt.ylabel('Voltage [ADC]')
for time in PETime:
    plt.vlines(time, minpoint, maxpoint, 'r')

plt.title("Entry %d, Event %d, Channel %d" % (entry, EventId, ChannelId))
plt.show()

h5file.close()