#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:26:56 2019

@author: xudachengthu

from https://ghost-hunter.net9.org/2019/02/01/submission.html

An example of writing a file
"""

# An example of writing a file.
import tables

# Define the database columns
class AnswerData(tables.IsDescription):
    EventID    = tables.Int64Col(pos=0)
    ChannelID  = tables.Int16Col(pos=1)
    PETime     = tables.Int16Col(pos=2)
    Weight     = tables.Float32Col(pos=3)

# Create the output file and the group
h5file = tables.open_file("MyAnswer.h5", mode="w", title="OneTonDetector")

# Create tables
AnswerTable = h5file.create_table("/", "Answer", AnswerData, "Answer")
answer = AnswerTable.row

# Write data 
answer['EventID'] =  1
answer['ChannelID'] = 0
answer['PETime'] = 269 
answer['Weight'] = 0.3 
answer.append()
answer['EventID'] =  1
answer['ChannelID'] = 0
answer['PETime'] = 284 
answer['Weight'] = 0.5 
answer.append()
answer['EventID'] =  1
answer['ChannelID'] = 0
answer['PETime'] = 287 
answer['Weight'] = 2.0 
answer.append()

# Flush into the output file
AnswerTable.flush()

h5file.close()