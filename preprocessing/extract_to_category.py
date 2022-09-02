import os
import numpy as np
def create_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_invert_dic(dic):
    inv_dic = {}
    for key in dic.keys():
        inv_dic[dic[key]] = key

    return inv_dic




def index_to_sentence(inv_dic, list_index):
    print([inv_dic[x]for x in list_index ])

target_path = 'four_category'
full_label = []
with open('E:/data/SER/IEMOCAP/processed/processed_label.txt') as f:
    full_label = f.readlines()
full_label = [x.strip() for x in full_label]
#print(full_label)

#id
import csv
tmp_ids = []
#label.csv [Ses01F_impro01_F000	neu]
with open('E:/data/SER/IEMOCAP/processed/label.csv') as f:
    csv_reader = csv.reader(f)
    tmp_ids = [x for x in csv_reader]
    tmp_ids = [x[0] for x in tmp_ids]
with open('E:/data/SER/IEMOCAP/processed/ordered_ids.txt', 'w') as f:
    for _id in tmp_ids:
        f.write(_id+'\n')
data = []
with open('E:/data/SER/IEMOCAP/processed/ordered_ids.txt') as f:
    data = f.readlines()
create_folder('E:/data/SER/IEMOCAP/processed/'+target_path)
with open('E:/data/SER/IEMOCAP/processed/' + target_path + '/FC_ordered_ids.txt', 'w') as f:
    for i, label in enumerate(full_label):
        if label != '-1':
            f.write( data[i] )
#labels
data = []
with open('E:/data/SER/IEMOCAP/processed/processed_label.txt') as f:
    data = f.readlines()
with open('E:/data/SER/IEMOCAP/processed/' + target_path + '/FC_label.txt', 'w') as f:
    for i, label in enumerate(full_label):
        if label != '-1':
            f.write( data[i] )

#MFCC
data = []
with open('E:/data/SER/IEMOCAP/processed/processed_MFCC12EDA_sequenceN.txt') as f:
    data = f.readlines()
# with open('../data/processed/IEMOCAP/four_category/FC_MFCC12EDAZ_sequenceN.txt', 'w') as f:
with open('E:/data/SER/IEMOCAP/processed/' + target_path + '/FC_MFCC12EDA_sequenceN.txt', 'w') as f:
    for i, label in enumerate(full_label):
        if label != '-1':
            print(i, label)
            print(len(data))
            f.write( data[i] )

#prodosy
data = np.load('E:/data/SER/IEMOCAP/processed/processed_prosody.npy')
np.shape(data)
total_num = 5531
new_data = np.zeros([5531, 35], dtype=float)
index = 0
for i, label in enumerate(full_label):
    if label != '-1':
        new_data[index] = data[i]
        index += 1
np.save('E:/data/SER/IEMOCAP/processed/' + target_path + '/FC_prodosy.npy', new_data)

#mfcc

data = np.load('E:/data/SER/IEMOCAP/processed/processed_MFCC12EDA.npy')
total_num = 5531
new_data = np.zeros([5531, 750, 39], dtype=np.float)
index = 0
for i, label in enumerate(full_label):
    if label != '-1':
        new_data[index] = data[i]
        index += 1
# np.save('../data/processed/IEMOCAP/four_category/FC_MFCC12EDAZ.npy', new_data)
np.save('E:/data/SER/IEMOCAP/processed/' + target_path + '/FC_MFCC12EDA.npy', new_data)

#transcription

data = []
with open('E:/data/SER/IEMOCAP/processed/processed_tran.csv') as f:
    csv_reader = csv.reader(f)
    data = [ x[1] for x in csv_reader ]

with open('E:/data/SER/IEMOCAP/processed/' + target_path + '/FC_trans.txt', 'w') as f:
    for i, label in enumerate(full_label):
        if label != '-1':
            f.write( data[i] + '\n')

#transcription numpy

import pickle

data = np.load('E:/data/SER/IEMOCAP/processed/processed_trans.npy')
#print(np.shape(data))
#print(len(full_label))
total_num = 5531
new_data = np.zeros([5531, 128], dtype=int)
index = 0
for i, label in enumerate(full_label):
    if label != '-1':
        new_data[index] = data[i]
        index += 1
np.save('E:/data/SER/IEMOCAP/processed/four_category/FC_trans.npy', new_data)

# let's test
dic = {}
with open('E:/data/SER/IEMOCAP/processed/dic.pkl', 'rb') as f:
    dic = pickle.load(f)
inv_dic = create_invert_dic(dic)
index_to_sentence(inv_dic, new_data[4999])

#test=np.load('E:/data/SER/IEMOCAP/processed/four_category/FC_trans.npy')
#print(test[0])