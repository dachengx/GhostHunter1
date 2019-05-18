#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 06:52:01 2019

@author: xudachengthu

batch

find out the Simple Features of ftraining-0.h5 -- MPES

find out the Simple Features of ftraining-0.h5 -- AEWL

find out the Simple Features of ftraining-0.h5 -- AEWH & AEWT

find out the Simple Features of ftraining-0.h5 -- MVR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

filename1 = "/Users/xudachengthu/Downloads/GHdataset/ftraining-0.h5"

playd = h5py.File(filename1)
ent = playd['Waveform']
answ = pd.read_hdf(filename1, "GroundTruth")

l = min(len(ent),10000)

print(l)
ent = ent[0:l]
answ = answ[0:20*l]
se = np.linspace(1, 20, 20, dtype='int32')
countall = np.asarray((se, np.zeros_like(se, dtype='int32'))).T

# information of magnitude of Voltage
minmaxvall = np.zeros([l,2], dtype = 'int16')
# information of the come and leave time of the wave
minmaxtall = np.zeros([l,3], dtype = 'int16')
# information of the magnitude of PE
minmaxpeall = np.zeros([l,4], dtype = 'int16')
averwf = np.zeros([l,1], dtype = 'float32')
# information of the hysteresis of the wave
distvpe = np.zeros([l,3], dtype = 'int16')
thres = 968
print(thres)
platnum = 976
print(platnum)
count = 0
for i in range(l):
    #print(i)
    count = count + 1
    if count == int(l / 100):
        print(int((i+1) / (l / 100)), end='% ', flush=True)
        count = 0
    eid = ent[i]['EventID']
    ch = ent[i]['ChannelID']
    pe = answ.query("EventID=={} & ChannelID=={}".format(eid, ch))
    pev = pe['PETime'].values
    u, c = np.unique(pev, return_counts=True)
    minipe = np.min(pev)
    maxipe = np.max(pev)
    minmaxpeall[i,0] = minipe
    minmaxpeall[i,1] = maxipe
    minmaxpeall[i,2] = maxipe - minipe
    minmaxpeall[i,3] = u.size
    unique, counts = np.unique(c,return_counts=True)
    for j in range(len(unique)):
        countall[unique[j]-1,1] = countall[unique[j]-1,1] + counts[j]
    #countall = np.concatenate((countall,np.asarray((unique,counts)).T))
    
    wf = ent[i]['Waveform']
    wf_valid = wf[200:606]
    af = np.where(wf_valid <= thres)
    if np.size(af) != 0:
        minit_v = af[0][0]
        tr = range(minit_v - 10, minit_v - 10 + 206)
    else:
        tr = range(0, 206)
    if minit_v - 10 + 206 < 406:
        miniv = np.min(wf_valid)
        maxiv = np.max(wf_valid)
        minmaxvall[i,0] = miniv
        minmaxvall[i,1] = maxiv
        wf_aver = np.mean(np.subtract(platnum, wf_valid[tr]))
        averwf[i,0] = wf_aver
    
    # thres can be changed
    #w = np.array(wf < thres, dtype=np.int8)
    #d = np.diff(w)
    # latter one subtract former one
    minit = 0
    maxit = 0
    af = np.where(wf[200:606] <= thres)
    if np.size(af) != 0:
        minit = af[0][0] + 200
        maxit = af[0][-1] + 200
        minmaxtall[i,0] = minit
        minmaxtall[i,1] = maxit
        minmaxtall[i,2] = maxit - minit
        dist1 = minit - minipe
        dist2 = maxit - maxipe
        if dist1 > 0:
            distvpe[i,0] = dist1
        if dist2 > 0:
            distvpe[i,1] = dist2
        if dist1 > 0 and dist2 > 0:
            distvpe[i,2] = maxit - minipe
#unique, counts = np.unique(countall[1:,1], return_counts=True)
#res = np.asarray((unique,counts)).T
#print(res.tolist())

print()
print(countall)

minv = np.min(minmaxvall[:,0])
maxv = np.max(minmaxvall[:,1])

mint = np.min(minmaxtall[:,0])
imint = np.argmin(minmaxtall[:,0])
maxt = np.max(minmaxtall[:,1])
imaxt = np.argmax(minmaxtall[:,1])
minlent = np.min(minmaxtall[:,2])
maxlent = np.max(minmaxtall[:,2])
meanlent = np.mean(minmaxtall[:,2])

minpe = np.min(minmaxpeall[:,0])
iminpe = np.argmin(minmaxpeall[:,0])
maxpe = np.max(minmaxpeall[:,1])
imaxpe = np.argmax(minmaxpeall[:,1])
minlenpe = np.min(minmaxpeall[:,2])
maxlenpe = np.max(minmaxpeall[:,2])
meanlenpe = np.mean(minmaxpeall[:,2])

distposi0 = distvpe[(distvpe[:,0]>0) & (distvpe[:,0]<10), 0]
distposi1 = distvpe[(distvpe[:,1]>0) & (distvpe[:,1]<10), 1]
distlong = distvpe[distvpe[:,2]>0, 2]
distlongest = np.max(distlong)
meandist = np.mean(distlong)

print(minv,maxv)
print(mint,imint,maxt,imaxt,minlent,maxlent,meanlent)
print(minpe,iminpe,maxpe,imaxpe,minlenpe,maxlenpe,meanlenpe)
print(distlongest,meandist)

plt.clf()
plt.hist(minmaxvall[:,0], 100, density=1, histtype='bar', cumulative=True)
plt.title('minvcumu, l='+str(l))
plt.savefig("minvcumu.eps")
plt.show()
plt.clf()
plt.hist(minmaxtall[:,2], 100, density=1, histtype='bar', cumulative=True)
plt.title('wavelencumu, l='+str(l))
plt.savefig("wavelencumu.eps")
plt.show()
plt.clf()
plt.hist(minmaxpeall[:,2], 100, density=1, histtype='bar', cumulative=True)
plt.title('pelencumu, l='+str(l))
plt.savefig("pelencumu.eps")
plt.show()
plt.clf()
plt.hist(minmaxpeall[:,3], 40, density=1, histtype='bar', cumulative=True)
plt.title('penumcumu, l='+str(l))
plt.savefig("penumcumu.eps")
plt.show()
plt.clf()
plt.hist(distlong, 50, density=1, histtype='bar', cumulative=True)
plt.title('totlencumu, l='+str(l))
plt.savefig("totlencumu.eps")
plt.show()
plt.clf()
plt.hist(distposi0, 50, density=1, histtype='bar', cumulative=True)
plt.title('distposi0cumu, l='+str(l))
plt.savefig("distposi0cumu.eps")
plt.show()
plt.clf()
plt.scatter(averwf[:,0]/100, minmaxpeall[:,3], c='r')
plt.title('corr of average_wf and pe_count, l='+str(l))
plt.savefig("corrwfpe.eps")
plt.show()
plt.clf()
plt.scatter(np.where(minmaxpeall[:,3] <= 30, minmaxpeall[:,3], 0), np.where(minmaxpeall[:,3] <= 30, averwf[:,0], 0), c='r')
plt.title('corr of average_wf and pe_count for pe_count <= 30, l='+str(l))
plt.savefig("corrwfpe30.eps")
plt.show()

penum_valid_raw = np.where(averwf[:,0]/100 < 0.3, minmaxpeall[:,3], 0)
averwf_valid_raw = np.where(averwf[:,0]/100 < 0.3, averwf[:,0]/100, 0)
penum_valid = penum_valid_raw[np.where(averwf_valid_raw != 0)]
averwf_valid = averwf_valid_raw[np.where(averwf_valid_raw != 0)]
plt.clf()
plt.scatter(averwf_valid, penum_valid, c='r')
plt.title('valid corr of wf and pe, l='+str(l))
plt.savefig("corrwfpe_valid.eps")
plt.show()

#reg = np.polyfit(averwf_valid, penum_valid, 2)
reg = np.divide(np.sum(np.multiply(averwf_valid, penum_valid)), np.sum(np.multiply(averwf_valid, averwf_valid)))
penum_pre = np.polyval([reg, 0], averwf_valid)
plt.clf()
plt.scatter(averwf_valid, penum_pre, c='r')
plt.savefig("corrwfpe_pre.eps")
plt.show()

playd.close()