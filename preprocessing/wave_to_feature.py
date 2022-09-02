#E:\data\SER\opensmile-config
import os
import sys
from util import file_search
import csv
import csv
import pandas as pd



OPENSMILE_CONFIG_PATH = 'E:/data/SER/opensmile-config/modified_MFCC12_E_D_A.conf'
#out_file = 'E:/data/SER/IEMOCAP/processed/MFCC12EDA.csv'
out_file = 'E:/data/SER/IEMOCAP/processed/MFCC12EDA_pre.csv'
lj='C:/Users/s123c/Desktop/opensmile-3.0-win-x64/bin'
os.chdir(lj)
'''
def extract_feature(list_in_file, out_file):
    cnt = 0
    #print("num",len(list_in_file))
    for in_file in list_in_file:
        out_file = 'E:/data/SER/IEMOCAP/processed/MFCC12EDA_pre.csv'
        test='E:/data/SER/IEMOCAP/Session1/sentences/wav/Ses01F_impro01\\Ses01F_impro01_F000.wav'
        #         cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -O ' + out_file + ' -headercsv 0'  #MFCC12EDAZ, prosody
        cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -csvoutput ' + out_file + ' -headercsv 0'  # MFCC12EDA
        os.system(cmd)
        csv_1 = pd.read_csv(out_file)
        csv_1.to_csv('E:/data/SER/IEMOCAP/processed/final_MFCC12EDA.csv',  mode='a', header=True,index=False)
        os.remove(out_file)
        cnt += 1
        print('now',cnt)

list_files = []

for x in range(2):
    sess_name = 'Session' + str(x + 1)
    path = 'E:/data/SER/IEMOCAP/' + sess_name + '/sentences/wav/'
    file_search(path, list_files)
    list_files = sorted(list_files)
    #print(list_files)
    #print(sess_name + ", #sum files: " + str(len(list_files)))
extract_feature(list_files, out_file)


with open('E:/data/SER/IEMOCAP/processed/final_MFCC12EDA.csv') as f:
    data = f.readlines()
print("hh",len(data))

#prosody
OPENSMILE_CONFIG_PATH = 'E:/data/SER/opensmile-config/modified_prosodyShsViterbiLoudness.conf'
out_file = 'E:/data/SER/IEMOCAP/processed/prosody.csv'

def extract_feature(list_in_file, out_file):

    cnt = 0
    for in_file in list_in_file:
        #cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -O ' + out_file + ' -headercsv 0'  # MFCC12EDAZ, prosody
        cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -csvoutput ' + out_file + ' -headercsv 0'   # MFCC12EDA
        os.system(cmd)
        cnt += 1
        if cnt % 1000 == 0:
            sys.stdout.flush()

list_files = []
for x in range(2):
    sess_name = 'Session' + str(x + 1)
    path = 'E:/data/SER/IEMOCAP/' + sess_name + '/sentences/wav/'
    file_search(path, list_files)
    list_files = sorted(list_files)
    print(sess_name + ", #sum files: " + str(len(list_files)))

extract_feature(list_files, out_file)
'''
#emobase
OPENSMILE_CONFIG_PATH = 'E:/data/SER/opensmile-config/modified_emobase2010.conf'
out_file = 'E:/data/SER/IEMOCAP/emobase.csv'

def extract_feature(list_in_file, out_file):

    cnt = 0
    for in_file in list_in_file:

        cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -csvoutput ' + out_file + ' -headercsv 0'   # MFCC12EDA
        #         cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -csvoutput ' + out_file + ' -headercsv 0'   # emobase2010
        os.system(cmd)
        cnt += 1
        if cnt % 1000 == 0:
            print
            str(cnt) + " / " + str(len(list_in_file))
            sys.stdout.flush()

list_files = []
for x in range(2):
    sess_name = 'Session' + str(x + 1)
    path = 'E:/data/SER/IEMOCAP/' + sess_name + '/sentences/wav/'
    file_search(path, list_files)
    list_files = sorted(list_files)
    print(sess_name + ", #sum files: " + str(len(list_files)))

extract_feature(list_files, out_file)


#load MFCC feature
import csv
#import cPickle
import numpy as np

lines = []

with open('E:/data/SER/IEMOCAP/processed/MFCC12EDA.csv') as f:
    csv_reader = csv.reader(f)
    lines = [x for x in csv_reader]
print(lines[1])
'''
data=pd.read_hdf('E:/data/SER/IEMOCAP/processed/MFCC12EDA.h5',key='data')
print(data)
for row_index, row in data.iterrows():
    if (row_index!=0):
        print(row_index,row[0])
        lines.append(row[0])
print("lines",len(lines))
'''
lines = [x[0].split(';') for x in lines[1:]]
#print(lines)
print("lines",len(lines))
float_lines = [[float(i) for i in x[1:]] for x in lines]  # do not care the first element
#print(float_lines)
# mark the index of each chunk

chunk_index = []
for i, line in enumerate(float_lines):
    if line[0] == 0:
        chunk_index.append(i)
print(chunk_index)
no_index_float_linex = [x[2:] for x in lines]  # remove first two element (sequence index)
#print(len(no_index_float_linex))

# merge sequence
list_MFCC = []
for i in range(len(chunk_index)):
    if i == len(chunk_index) - 1:  # last case
        list_MFCC.append(no_index_float_linex[chunk_index[i]:])
    else:
        list_MFCC.append(no_index_float_linex[chunk_index[i]:chunk_index[i + 1]])
#print(list_MFCC)
stat = [ len(x) for x in list_MFCC ]
print(np.mean(stat))
print (np.std(stat))
print (np.max(stat))
print (np.min(stat))
np.save('E:/data/SER/IEMOCAP/processed/processed_MFCC12EDA_sequenceN.npy', np.asarray(stat))

with open('E:/data/SER/IEMOCAP/processed/processed_MFCC12EDA_sequenceN.txt', 'w') as f:
    for data in stat:
        f.write( str(data) + '\n' )
np_MFCC = np.zeros( [10039, 750, 39], dtype=float)
for i in range(len(list_MFCC)):
    if len(list_MFCC[i]) > 750:
        np_MFCC[i][:] = list_MFCC[i][:750]
    else:
        np_MFCC[i][:len(list_MFCC[i])] = list_MFCC[i][:]
np.save('E:/data/SER/IEMOCAP/processed/processed_MFCC12EDA.npy', np_MFCC)

#%%
#prosody
lines = []
with open('E:/data/SER/IEMOCAP/processed/prosody.csv') as f:
    csv_reader = csv.reader(f)
    lines = [x for x in csv_reader]

#print(lines[0][1:-1])
np_prosody = np.zeros([10039, 35], dtype=np.float)
for i in range(len(np_prosody)):
    np_prosody[i] = lines[i][1:-1]
np.save('E:/data/SER/IEMOCAP/processed/processed_prosody.npy', np_prosody)
'''
#emobase

lines = []
with open('E:/data/SER/IEMOCAP/processed/emobase.csv') as f:
    csv_reader = csv.reader(f)
    lines = [x for x in csv_reader]

#print("exa",lines[0])
np_emobase = np.zeros([10039, 1582], dtype=np.float)
np.shape(np_emobase)
for i in range(len(np_emobase)):
    np_emobase[i] = lines[i][1:]

np.save('E:/data/SER/IEMOCAP/processed/processed_emobase.npy', np_emobase)
tmp = np.load('E:/data/SER/IEMOCAP/processed/processed_emobase.npy')
#print(np.shape(tmp))
'''