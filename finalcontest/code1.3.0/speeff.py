#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 19:47:03 2019

@author: xudachengthu

Using "single PE' method the generate the answer
"""

import numpy as np
import h5py
import time
import standard
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
'''
fipt = "/Users/xudachengthu/Downloads/GHdataset/finalcontest_data/ztraining-0.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/submission/first-submission-spe-fin.h5"


fipt = "/Users/xudachengthu/Downloads/GHdataset/finalcontest_data/zincm-problem.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/submission/first-submission-spe-fin.h5"
'''
fipt = "/Users/xudachengthu/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/Users/xudachengthu/Downloads/GHdataset/playground/first-submission-spe.h5"


fipt = "/home/xudacheng/Downloads/GHdataset/finalcontest_data/zincm-problem.h5"
fopt = "/home/xudacheng/Downloads/GHdataset/submission/first-submission-spe-fin.h5"
'''
fipt = "/home/xudacheng/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/home/xudacheng/Downloads/GHdataset/playground/first-submission-spe.h5"
'''
#LEARNING_RATE = 0.005
STEPS = 10000
#Length_pe = 200
Length_pe = 1029
#THRES = 968
#NORMAL_P = 30
BATCH_SIZE = 100

KNIFE = 0.05

AXE = 4
AXE_SWITCH = 1

EXP = 2

FILTER = 0

SHOWS = 0

def generate_eff_ft():
    opd = [('EventID', '<i8'), ('ChannelID', '<i2'), ('PETime', 'f4'), ('Weight', 'f4')]
    model = generate_model(standard.single_pe_path)
    
    plt.clf()
    plt.plot(model[0 : 50])
    plt.show()
    
    model = np.where(model > AXE, model - AXE, 0)
    
    model_raw = np.concatenate([model, np.zeros(Length_pe - len(model))])
    
    core = np.square(model / np.max(model))
    model_plate = np.ones_like(model)
    for i in range(EXP):
        model_plate = model_plate * core
    model = model_plate * np.max(model)
    #model = np.where(model > 0.02, model, 0)
    
    plt.clf()
    plt.plot(model[0 : 50])
    plt.show()
    
    model_ame = np.concatenate([model, np.zeros(Length_pe - len(model) + 200)])
    
    model_k = fft(model_ame)
    
    #tt = ifft(model_k)
    
    mtray = np.concatenate((np.zeros(Length_pe), model_raw[0 : 50], np.zeros(Length_pe)))
    loperator = np.concatenate([mtray[Length_pe - i : 2 * Length_pe + 50 - i] for i in range(Length_pe)]).reshape(Length_pe, Length_pe + 50)
    
    with h5py.File(fipt, 'r') as ipt, h5py.File(fopt, "w") as opt:
        ent = ipt['Waveform']
        l = len(ent)
        #l = 70
        print(l)
        dt = np.zeros(l*Length_pe, dtype = opd)
        start = 0
        end = 0
        count = 0
        
        for i in range(l):
            #i = 12134
            wf_input = ent[i]['Waveform']
            #wf_input = - model_raw
            #wf_input = np.concatenate([ent[228023]['Waveform'][0:500], ent[291379]['Waveform'][0:529]])
            
            fringe = np.zeros(100)
            wf_input = np.subtract(np.mean(wf_input[900:1000]), wf_input)
            wf_input = np.where(wf_input > 0, wf_input, 0)
            
            wf_input = np.where(wf_input > AXE, wf_input - AXE, 0)
            
            wf_input = np.concatenate([fringe, wf_input, fringe])
            
            wf_k = fft(wf_input)
            
            wf_k[0 : FILTER] = 0
            wf_k[len(wf_k) - FILTER : len(wf_k)] = 0
            
            #k_r = wf_k.real
            #k_r = np.where(np.abs(k_r) > 0.1, k_r, 0)
            spec = np.divide(wf_k, model_k)
            
            pf = ifft(spec)
            pf = pf.real
            #pf = np.divide(pf, Length_pe)
            pf = pf[100 : Length_pe + 100]
            
            if SHOWS:
                plt.clf()
                plt.title("wf_input")
                plt.plot(wf_input)
                plt.show()
                
                plt.clf()
                plt.title("ifft(wf_k)")
                plt.plot(ifft(wf_k))
                plt.show()
                
                plt.clf()
                #plt.plot(wf_k.real[100 : Length_pe + 100])
                #plt.plot(wf_k.real)
                #plt.plot(model_k.real[100 : Length_pe + 100])
                plt.title("pf")
                plt.plot(pf)
                plt.show()
                
                a = np.matmul(pf, loperator)
                #a = np.matmul(pf, loperator)
                plt.clf()
                #plt.plot(a[250 : 400])
                plt.title("np.matmul(pf, loperator)")
                plt.plot(a)
                plt.show()
                
                a = np.matmul(np.where(pf > 0, pf, 0), loperator)
                #a = np.matmul(pf, loperator)
                plt.clf()
                #plt.plot(a[250 : 400])
                plt.title("np.matmul(np.where(pf > 0, pf, 0), loperator)")
                plt.plot(a)
                plt.show()
                
            
            pf = np.where(pf > KNIFE, pf, 0)
            
            lenpf = np.size(np.where(pf > 0))
            if lenpf == 0:
                pf[300] = 1
            lenpf = np.size(np.where(pf > 0))
            pet = np.where(pf > 0)[0]
            pwe = pf[pf > 0]
            end = start + lenpf
            dt['PETime'][start:end] = pet
            dt['Weight'][start:end] = pwe
            dt['EventID'][start:end] = ent[i]['EventID']
            dt['ChannelID'][start:end] = ent[i]['ChannelID']
            start = end
                    
            count = count + 1
            if count == int(l / 100) + 1:
                print(int((i+1) / (l / 100)), end = '% ', flush=True)
                count = 0
            
        dt = dt[np.where(dt['Weight'] > 0)]
        opt.create_dataset('Answer', data = dt, compression='gzip')
        print(fopt, end = ' ', flush=True)

def generate_model(spe_path):
    speFile = h5py.File(spe_path, 'r')
    spemean = np.mean(speFile['Sketchy']['speWf'], axis = 0)
    base_vol = np.mean(spemean[70:120])
    stdmodel = np.subtract(base_vol, spemean[20:120])
    #stdmodel = np.multiply(np.around(np.divide(stdmodel, 0.05)), 0.05)
    stdmodel = np.where(stdmodel > 0.02, stdmodel, 0)
    
    stdmodel = np.abs(np.where(stdmodel >= 0, stdmodel, 0))
    
    speFile.close()
    return stdmodel

def main():
    start_t = time.time()
    generate_eff_ft()
    end_t = time.time()
    print(end_t - start_t)

if __name__ == '__main__':
    main()