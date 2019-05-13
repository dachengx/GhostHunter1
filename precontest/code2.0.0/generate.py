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

h5_file_path = '/Users/xudachengthu/Downloads/GHdataset/ftraining-0.h5'
tfRecord_train = '/Users/xudachengthu/Downloads/GHdataset/tfrecorddata/h5_train.tfrecords'
tfRecord_test = '/Users/xudachengthu/Downloads/GHdataset/tfrecorddata/h5_test.tfrecords'
data_path = '/Users/xudachengthu/Downloads/GHdataset/tfrecorddata'
Length_waveform = 1029

def write_tfRecord(tfRecordName, h5_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    h5file = h5py.File(h5_path)
    ent = h5file['Waveform']
    answ = pd.read_hdf(h5_path, "GroundTruth")
    #lenwf = len(ent[0]['Waveform'])
    lenwf = Length_waveform
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
        unipe = np.unique(pev, return_counts=False).tolist()
        wf = ent[i]['Waveform'].tolist()
        pet = [0] * lenwf
        for j in unipe:
            if j < lenwf:
                pet[j-1] = 1
        
        example = tf.train.Example(features = tf.train.Features(feature={
                'waveform': tf.train.Feature(int64_list=tf.train.Int64List(value=wf)), 
                'petime': tf.train.Feature(int64_list=tf.train.Int64List(value=pet))}))
        writer.write(example.SerializeToString())
        count = count + 1
        if count == int(l / 100)+1:
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
                                               'petime': tf.FixedLenFeature([Length_waveform], tf.int64)})
    wf = tf.cast(features['waveform'], tf.float32) * (1./1000)
    pet = tf.cast(features['petime'], tf.float32)
    return wf, pet

def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    wf, pet = read_tfRecord(tfRecord_path)
    wf_batch, pet_batch = tf.train.shuffle_batch([wf, pet], 
                                                 batch_size = num, 
                                                 num_threads = 2, 
                                                 capacity = 1000, 
                                                 min_after_dequeue = 700)
    return wf_batch, pet_batch

def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print('The directory was created successfully')
    else:
        print('Directory already exists')
    write_tfRecord(tfRecord_train, h5_file_path)
    write_tfRecord(tfRecord_test, h5_file_path)

def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()