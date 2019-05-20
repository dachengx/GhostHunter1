***

#### This is the first edit. 

Using gradient descent and fully connected network. 
Directly input all the 1029 signals to the NN, and output the probability of existence of PE in each 1029 signal sampling point. 

The data used in generating tfRecord_train and tfRecord_test are the same! 

Pay attention to the 'correct_prediction' and 'loss'

Predict "True" if y[i] > 0.5, otherwise, predict "False"

#### This is the second edit

#### This is the third edit

Only use waveform data in [201,600]

#### This is the 4th edit

STILL using tfrecords files from 2.1.1

change relu => sigmoid

Add convolutional method

#### This is the 5th edit

Only train parameters for the 300th waveform position

*DO NOT* change the process.py

#### This is the 6th edit
ADD convolutional method