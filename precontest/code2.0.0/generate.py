#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:58:19 2019

@author: xudachengthu

generate tf.records files which can be used by tensorflow

/Users/xudachengthu/Downloads/GHdataset/ftraining-0.h5 --the file downloaded from crowdAI
"""

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import h5py

filename1 = "/Users/xudachengthu/Downloads/GHdataset/ftraining-0.h5"

playd = h5py.File(filename1)
ent = playd['Waveform']
answ = pd.read_hdf(filename1, "GroundTruth")

l = min(len(ent),1000)

print(l)
ent = ent[0:l]
answ = answ[0:20*l]