#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:58:19 2019

@author: xudachengthu

generate tf.records files which can be used by tensorflow

/Users/xudachengthu/Downloads/GHdataset/ftraining-0.h5 --the file downloaded from crowdAI
"""

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os

h5_train_path = '/Users/xudachengthu/Downloads/GHdataset/ftraining-0.h5'
h5_test_path = '/Users/xudachengthu/Downloads/GHdataset/ftraining-1.h5'
tfRecord_train = '/Users/xudachengthu/Downloads/GHdataset/tfrecorddata/h5_train_2.3.2.tfrecords'
tfRecord_test = '/Users/xudachengthu/Downloads/GHdataset/tfrecorddata/h5_test_2.3.2.tfrecords'
data_path = '/Users/xudachengthu/Downloads/GHdataset/tfrecorddata'
#Length_waveform = 400
Length_waveform = 206
#Length_pestate = 2
THRES = 968
PLATNUM = 976

def write_tfRecord(tfRecordName, h5_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    h5file = h5py.File(h5_path)
    ent = h5file['Waveform']
    answ = pd.read_hdf(h5_path, "GroundTruth")
    #lenwf = len(ent[0]['Waveform'])
    #lenwf = Length_waveform
    l = min(len(ent),10000)
    print(l)
    ent = ent[0:l]
    answ = answ[0:20*l]
    count = 0
    for i in range(l):
        eid = ent[i]['EventID']
        ch = ent[i]['ChannelID']
        pe = answ.query("EventID=={} & ChannelID=={}".format(eid, ch))
        pev = pe['PETime'].values
        unipe = np.unique(pev, return_counts=False)
        wf = ent[i]['Waveform']
        af = np.where(wf[200:606] <= THRES)
        if np.size(af) != 0:
            minit_v = af[0][0]
            tr = range(minit_v - 10 + 200, minit_v - 10 + Length_waveform + 200)
            wf_test = wf[tr]
            pet = [0] * Length_waveform
            unipe = unipe[(unipe >= tr[0]) & (unipe < tr[-1])] - tr[0]
            wf_aver = np.mean(np.subtract(PLATNUM, wf_test)) * (1./100)
            for j in unipe:
                pet[j-1] = 1
        
            example = tf.train.Example(features = tf.train.Features(feature={
                'waveform': tf.train.Feature(int64_list=tf.train.Int64List(value=wf_test)), 
                'petime': tf.train.Feature(int64_list=tf.train.Int64List(value=pet)), 
                'averwf': tf.train.Feature(float_list=tf.train.FloatList(value=[wf_aver]))}))
            writer.write(example.SerializeToString())
        
        count = count + 1
        if count == int(l / 100) + 1:
            print(int((i+1) / (l / 100)), end='% ')
            count = 0
    h5file.close()
    writer.close()
    print()
    print('Write tfrecord successfully to ' + tfRecordName)

def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, 
                                       features={
                                               'waveform': tf.FixedLenFeature([Length_waveform], tf.int64), 
                                               'petime': tf.FixedLenFeature([Length_waveform], tf.int64), 
                                               'averwf': tf.FixedLenFeature([1], tf.float32)})
    '''!!!'''
    wf = tf.cast(features['waveform'], tf.float32) * (1./1000)
    pet = tf.cast(features['petime'], tf.float32)
    aver = tf.cast(features['averwf'], tf.float32)
    return wf, pet, aver

def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    wf, pet, aver = read_tfRecord(tfRecord_path)
    wf_batch, pet_batch, aver_batch = tf.train.shuffle_batch([wf, pet, aver], 
                                                 batch_size = num, 
                                                 num_threads = 2, 
                                                 capacity = 1000, 
                                                 min_after_dequeue = 700)
    return wf_batch, pet_batch, aver_batch

def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print('The directory was created successfully')
    else:
        print(data_path + ' Directory already exists')
    isExists = os.path.exists(tfRecord_train)
    if not isExists:
        write_tfRecord(tfRecord_train, h5_train_path)
    else:
        print(tfRecord_train + ' already exists')
    isExists = os.path.exists(tfRecord_test)
    if not isExists:
        write_tfRecord(tfRecord_test, h5_test_path)
    else:
        print(tfRecord_test + ' already exists')

def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()