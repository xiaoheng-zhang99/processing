#in this file, we will transform each sentence into word_id
import os
import sys
import csv
import pickle
import numpy as np
#from nlp_util import *
from nltk.tokenize import word_tokenize

lines = []
with open('E:/data/SER/IEMOCAP/processed/processed_tran.csv') as f:
    # with open('../data/processed/IEMOCAP/processed_tran_fromG.csv') as f:
    read = csv.reader(f)
    lines = [x[1] for x in read]

token_lines = [word_tokenize(x) for x in lines]
token_lines_lower = [[t.lower() for t in x] for x in token_lines]


def read_data(dic, lines):
    for tokens in lines:
        for token in tokens:
            token = token.lower()
            if token in dic:
                dic[token] += 1
            else:
                dic[token] = 1
    return dic

dic_count = {}
dic_count = read_data(dic_count, token_lines_lower)
#print(dic_count)
print('dic size : ' + str(len(dic_count)))

dic = {}
dic['_PAD_'] = len(dic)
dic['_UNK_'] = len(dic)

for word in dic_count.keys():
    dic[word] = len(dic)
print(len(dic))#3736
#print(dic['_PAD_'])#0


with open('E:/data/SER/IEMOCAP/processed/dic.pkl', 'wb') as f:
# with open('../data/processed/IEMOCAP/dic_G.pkl', 'w') as f:
    pickle.dump(dic,f)

lines = []
with open('E:/data/SER/IEMOCAP/processed/processed_tran.csv') as f:
    # with open('../data/processed/IEMOCAP/processed_tran_fromG.csv') as f:
    read = csv.reader(f)
    lines = [x[1] for x in read if (x!=[])]

token_lines = [word_tokenize(x) for x in lines]
token_lines_lower = [[t.lower() for t in x] for x in token_lines]
#sent_len = [len(x) for x in token_lines]
index_lines = [ [ dic[t] for t in x ] for x in token_lines_lower ]
print(len(index_lines))
#save as npy file
np_trans = np.zeros( [3630, 128], dtype=int)
np.shape(np_trans)
for i in range(len(index_lines)):
    if len(index_lines[i]) > 127:
        np_trans[i][:] = index_lines[i][:128]
    else:
        np_trans[i][:len(index_lines[i])] = index_lines[i][:]
np.save('E:/data/SER/IEMOCAP/processed/processed_trans.npy', np_trans)
np.shape(np_trans)
#test=np.load('E:/data/SER/IEMOCAP/processed/processed_trans.npy')
#print(test[:10])