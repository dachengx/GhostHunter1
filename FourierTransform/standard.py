#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:51:47 2019

@author: xudachengthu

Generate standard response model of single pe's waveform
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import time

def generate_standard(h5_path, single_pe_path):
    

def main():
    start_t = time.time()
    h5_path = '/home/xudacheng/Downloads/GHdataset/finalcontest_data/ztraining-0.h5'
    single_pe_path = '/home/xudacheng/Downloads/GHdataset/sketchystore/single_pe.h5'
    generate_standard(h5_path, single_pe_path) # generate response model
    end_t = time.time()
    print(end_t - start_t)

if __name__ == '__main__':
    main()