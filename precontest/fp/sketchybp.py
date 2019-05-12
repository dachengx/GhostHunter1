#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:19:46 2019

@author: xudachengthu

batch

find out the Simple Features of first-problem.h5 -- AEWL

find out the Simple Features of first-problem.h5 -- AEWH & AEWT

find out the Simple Features of first-problem.h5 -- MVR
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, h5py

filename1 = "/Users/xudachengthu/Downloads/GHdataset/first-problem.h5"

playd = h5py.File(filename1)
ent = playd['Waveform']
#answ = pd.read_hdf(filename1, "GroundTruth")

l = min(len(ent),100000)

print(l)
ent = ent[0:l]
#answ = answ[0:20*l]
se = np.linspace(1, 20, 20, dtype='int32')
#countall = np.asarray((se, np.zeros_like(se, dtype='int32'))).T

# information of magnitude of Voltage
minmaxvall = np.zeros([l,2], dtype = 'int16')
# information of the come and leave time of the wave
minmaxtall = np.zeros([l,3], dtype = 'int16')
'''
# information of the magnitude of PE
minmaxpeall = np.zeros([l,3], dtype = 'int16')
# information of the hysteresis of the wave
distvpe = np.zeros([l,3], dtype = 'int16')
'''
thres = 968
print(thres)
count = 0
for i in range(l):
    #print(i)
    count = count + 1
    if count == int(l / 100):
        print(int((i+1) / (l / 100)), end='% ')
        count = 0
    '''
    eid = ent[i]['EventID']
    ch = ent[i]['ChannelID']
    pe = answ.query("EventID=={} & ChannelID=={}".format(eid, ch))
    pev = pe['PETime'].values
    _, c = np.unique(pev, return_counts=True)
    minipe = np.min(pev)
    maxipe = np.max(pev)
    minmaxpeall[i,0] = minipe
    minmaxpeall[i,1] = maxipe
    minmaxpeall[i,2] = maxipe - minipe
    unique, counts = np.unique(c,return_counts=True)
    for j in range(len(unique)):
        countall[unique[j]-1,1] = countall[unique[j]-1,1] + counts[j]
    #countall = np.concatenate((countall,np.asarray((unique,counts)).T))
    '''
    
    wf = ent[i]['Waveform']
    miniv = np.min(wf)
    maxiv = np.max(wf)
    minmaxvall[i,0] = miniv
    minmaxvall[i,1] = maxiv
    
    # thres can be changed
    w = np.array(wf < thres, dtype=np.int8)
    d = np.diff(w)
    # latter one subtract former one
    if d.max() != 0 and d.min() !=0:
        minit = np.where(d >= 1)[0][0]
        maxit = np.where(d <= -1)[0][-1]
        minmaxtall[i,0] = minit
        minmaxtall[i,1] = maxit
        minmaxtall[i,2] = maxit - minit
    
    '''
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
        '''

print()
#print(countall)

minv = np.min(minmaxvall[:,0])
maxv = np.max(minmaxvall[:,1])

mint = np.min(minmaxtall[:,0])
imint = np.argmin(minmaxtall[:,0])
maxt = np.max(minmaxtall[:,1])
imaxt = np.argmax(minmaxtall[:,1])
minlent = np.min(minmaxtall[:,2])
maxlent = np.max(minmaxtall[:,2])
meanlent = np.mean(minmaxtall[:,2])

'''
minpe = np.min(minmaxpeall[:,0])
iminpe = np.argmin(minmaxpeall[:,0])
maxpe = np.max(minmaxpeall[:,1])
imaxpe = np.argmax(minmaxpeall[:,1])
minlenpe = np.min(minmaxpeall[:,2])
maxlenpe = np.max(minmaxpeall[:,2])
meanlenpe = np.mean(minmaxpeall[:,2])

distposi0 = distvpe[distvpe[:,0]>0, 0]
distposi1 = distvpe[distvpe[:,1]>0, 1]
distlong = distvpe[distvpe[:,2]>0, 2]
distlongest = np.max(distlong)
meandist = np.mean(distlong)
'''

print(minv,maxv)
print(mint,imint,maxt,imaxt,minlent,maxlent,meanlent)
#print(minpe,iminpe,maxpe,imaxpe,minlenpe,maxlenpe,meanlenpe)
#print(distlongest,meandist)

plt.clf()
plt.hist(minmaxvall[:,0], 100, density=1, histtype='bar', cumulative=True)
plt.title('minvcumu, l='+str(l))
plt.savefig("minvcumu.eps")
plt.show()
plt.clf()
plt.hist(minmaxtall[:,2], 100, density=1, histtype='bar', cumulative=True)
plt.title('wavelenumu, l='+str(l))
plt.savefig("wavelenumu.eps")
plt.show()
'''
plt.clf()
plt.hist(minmaxpeall[:,2], 100, density=1, histtype='bar', cumulative=True)
plt.title('pelenumu, l='+str(l))
plt.savefig("pelenumu.eps")
plt.show()
plt.clf()
plt.hist(distlong, 50, density=1, histtype='bar', cumulative=True)
plt.title('totlenumu, l='+str(l))
plt.savefig("totlenumu.eps")
plt.show()
'''

playd.close()