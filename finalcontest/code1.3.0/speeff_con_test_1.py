#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 12:07:33 2019

@author: xudachengthu
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

fopt_prefix = "/Users/xudachengthu/Downloads/GHdataset/playground/"

'''
fipt = "/home/xudacheng/Downloads/GHdataset/finalcontest_data/zincm-problem.h5"
fopt = "/home/xudacheng/Downloads/GHdataset/submission/first-submission-spe-fin.h5"

fipt = "/home/xudacheng/Downloads/GHdataset/playground/playground-data.h5"
fopt = "/home/xudacheng/Downloads/GHdataset/playground/first-submission-spe.h5"

fopt_prefix = "/home/xudacheng/Downloads/GHdataset/playground/"
'''
Length_pe = 1029

KNIFE = [0.01, 0.03, 0.1]

AXE = [4]

EXP = [2, 3, 4, 5, 6]

FILTER = [0, 1, 2]

SHOWS = 0

def generate_eff_ft(knife, axe, exp, filte_r, fopt):
    opd = [('EventID', '<i8'), ('ChannelID', '<i2'), ('PETime', 'f4'), ('Weight', 'f4')]
    model = generate_model(standard.single_pe_path)
    
    plt.clf()
    plt.plot(model[0 : 50])
    plt.show()
    
    model = np.where(model > axe, model - axe, 0)
    
    model_raw = np.concatenate([model, np.zeros(Length_pe - len(model))])
    
    core = np.square(model / np.max(model))
    model_plate = np.ones_like(model)
    for i in range(exp):
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
            
            wf_input = np.where(wf_input > axe, wf_input - axe, 0)
            
            wf_input = np.concatenate([fringe, wf_input, fringe])
            
            wf_k = fft(wf_input)
            
            wf_k[0 : filte_r] = 0
            wf_k[len(wf_k) - filte_r : len(wf_k)] = 0
            
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
                plt.plot(ifft(wf_k)[200:450])
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
                
            
            pf = np.where(pf > knife, pf, 0)
            
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
    for i in range(len(KNIFE)):
        for j in range(len(AXE)):
            for m in range(len(EXP)):
                for n in range(len(FILTER)):
                    fopt = fopt_prefix + str(i) + '-' + str(j) + '-' + str(m) + '-' + str(n) + '.h5'
                    start_t = time.time()
                    generate_eff_ft(KNIFE[i], AXE[j], EXP[m], FILTER[n], fopt)
                    end_t = time.time()
                    print('Time for ' + str(KNIFE[i]) + ' ' + str(AXE[j]) + ' ' + str(EXP[m]) + ' ' + str(FILTER[n]) + ' is ' + str(end_t - start_t))

if __name__ == '__main__':
    main()