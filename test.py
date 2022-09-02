import pandas as pd
import numpy as np
from data_loader import *

'''
data = []
with open('E:/data/SER/IEMOCAP/processed/MFCC12EDA.csv', 'r', encoding='gbk', errors='ignore') as f:
    for line in f:
        data.append(line.split(','))

data = pd.DataFrame(data)

h5 = pd.HDFStore('E:/data/SER/IEMOCAP/processed/MFCC12EDA.h5','w', complevel=4, complib='blosc')
h5['data'] = data.astype('int64')
h5.close()

data=pd.read_hdf('E:/data/SER/IEMOCAP/processed/emobase.h5',key='data')
for row_index, row in data.iterrows():
    #print(row)
    print(row[0])

batch_gen=ProcessDataText('E:\data\SER\IEMOCAP\processed\\four_category\\audio_woZ_set01\\')
#f=np.load('E:\data\SER\IEMOCAP\processed\\four_category\\audio_woZ_set01\\test_label.npy')
#print(f)
batch_gen.get_batch(
        data=batch_gen.train_set,
        batch_size=3,
        encoder_size=128,
        is_test=False
    )

import tensorflow as tf

state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

with tf.Session() as sess:
    print(sess.run(tf.global_variables_initializer()))
    for _ in range(3):
        print(sess.run(update))
        print(sess.run(state))
'''

batch_gen=ProcessDataMulti('E:\data\SER\IEMOCAP\processed\\four_category\\audio_woZ_set01\\')
#f=np.load('E:\data\SER\IEMOCAP\processed\\four_category\\audio_woZ_set01\\test_label.npy')
#print(f)
encoder_inputs_audio, encoder_seq_audio, encoder_prosody, encoder_inputs_text, encoder_seq_text,_=batch_gen.get_batch(
        data=batch_gen.dev_set,
        batch_size=3,
        encoder_size_audio=750,
        encoder_size_text=128,
        is_test=False)
print(len(encoder_inputs_audio[0]))