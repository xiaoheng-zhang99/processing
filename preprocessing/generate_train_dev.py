target_path_name = 'four_category'
import random

lines = []
with open('E:/data/SER/IEMOCAP/processed/' + target_path_name + '/FC_label.txt') as f :
    lines = f.readlines()

random.seed(337)
cfold01_test = []
cfold02_test = []
cfold03_test = []
cfold04_test = []
cfold05_test = []

for x in range(len(lines)):

    compare = random.random()

    if compare > 0.8:
        cfold01_test.append(str(x))
    elif compare > 0.6:
        cfold02_test.append(str(x))
    elif compare > 0.4:
        cfold03_test.append(str(x))
    elif compare > 0.2:
        cfold04_test.append(str(x))
    else:
        cfold05_test.append(str(x))

#print(cfold01_test)
def gen_data(list_all, list_test, path):
    random.seed(3372)

    train = []
    dev = []
    test = []

    for index in range(len(list_all)):
        compare = random.random()

        if str(index) not in list_test:
            train.append(str(index))
        else:
            if compare > 0.70:
                dev.append(str(index))
            else:
                test.append(str(index))

    with open(path, 'w') as f:
        f.write(' '.join(train) + '\n')
        f.write(' '.join(dev) + '\n')
        f.write(' '.join(test) + '\n')
    print("train",len(train))
    print("dev",len(dev))
    print("test",len(test))

from preprocessing.util import *
create_folder('E:/data/SER/IEMOCAP/processed/'+target_path_name+'/audio_woZ_set01')
create_folder('E:/data/SER/IEMOCAP/processed/'+target_path_name+'/audio_woZ_set02')
create_folder('E:/data/SER/IEMOCAP/processed/'+target_path_name+'/audio_woZ_set03')
create_folder('E:/data/SER/IEMOCAP/processed/'+target_path_name+'/audio_woZ_set04')
create_folder('E:/data/SER/IEMOCAP/processed/'+target_path_name+'/audio_woZ_set05')

path = 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/audio_woZ_set01/audio_woZ_set01.txt'
gen_data( lines, cfold01_test, path )
path = 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/audio_woZ_set02/audio_woZ_set02.txt'
gen_data( lines, cfold02_test, path )
path = 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/audio_woZ_set03/audio_woZ_set03.txt'
gen_data( lines, cfold03_test, path )
path = 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/audio_woZ_set04/audio_woZ_set04.txt'
gen_data( lines, cfold04_test, path )
path = 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/audio_woZ_set05/audio_woZ_set05.txt'
gen_data( lines, cfold05_test, path )

#generate data according to ids
import numpy as np

def extract_data_with_ids( npy_data, ids ) :
    npy_data_select = npy_data[ids][:][:]
    print (np.shape(npy_data_select))
    return npy_data_select


target_name_list=['audio_woZ_set01','audio_woZ_set01','audio_woZ_set01','audio_woZ_set04','audio_woZ_set05']

def generate_train_test(target_name):
    cmd = 'mklink /D  E:/data/SER/IEMOCAP/processed/four_category/'+target_name+'/ E:/data/SER/IEMOCAP/processed/W_embedding.npy'
    os.system(cmd)

    cmd = 'mklink /D  E:/data/SER/IEMOCAP/processed/four_category/'+target_name+'/ E:/data/SER/IEMOCAP/processed/dic.pkl'
    os.system(cmd)

    target_sequence = 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/' + target_name + '/' + target_name + '.txt'
    target_path = 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/' + target_name + '/'
    train_ids = []
    dev_ids = []
    test_ids = []
    with open( target_sequence ) as f:
        lines = f.readlines()
        train_ids = [ int(x) for x in lines[0].strip().split(' ')]
        dev_ids =  [ int(x) for x in lines[1].strip().split(' ')]
        test_ids = [ int(x) for x in lines[2].strip().split(' ')]

    print (len(train_ids))
    print (len(dev_ids))
    print (len(test_ids))


    # MFCC
    train_audio_mfcc = extract_data_with_ids( np.load( 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/FC_MFCC12EDA.npy' ), train_ids  )
    dev_audio_mfcc  = extract_data_with_ids( np.load( 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/FC_MFCC12EDA.npy' ), dev_ids  )
    test_audio_mfcc  = extract_data_with_ids( np.load( 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/FC_MFCC12EDA.npy' ), test_ids  )
    
    np.save( target_path + 'train_audio_mfcc.npy', train_audio_mfcc)
    np.save( target_path + 'dev_audio_mfcc.npy', dev_audio_mfcc)
    np.save( target_path + 'test_audio_mfcc.npy', test_audio_mfcc)

    # prosody
    train_audio_prosody = extract_data_with_ids(
        np.load('E:/data/SER/IEMOCAP/processed/' + target_path_name + '/FC_prodosy.npy'), train_ids)
    dev_audio_prosody = extract_data_with_ids(np.load('E:/data/SER/IEMOCAP/processed/' + target_path_name + '/FC_prodosy.npy'),
                                              dev_ids)
    test_audio_prosody = extract_data_with_ids(np.load('E:/data/SER/IEMOCAP/processed/' + target_path_name + '/FC_prodosy.npy'),
                                               test_ids)

    np.save(target_path + 'train_audio_prosody.npy', train_audio_prosody)
    np.save(target_path + 'dev_audio_prosody.npy', dev_audio_prosody)
    np.save(target_path + 'test_audio_prosody.npy', test_audio_prosody)

    # emobase2010
    # train_audio_emobase2010 = extract_data_with_ids( np.load( '../data/processed/IEMOCAP/' + target_path_name + '/FC_emobase2010.npy' ), train_ids  )
    # dev_audio_emobase2010  = extract_data_with_ids( np.load( '../data/processed/IEMOCAP/' + target_path_name + '/FC_emobase2010.npy' ), dev_ids  )
    # test_audio_emobase2010  = extract_data_with_ids( np.load( '../data/processed/IEMOCAP/' + target_path_name + '/FC_emobase2010.npy' ), test_ids  )

    # np.save( target_path + 'train_audio_emobase2010.npy', train_audio_emobase2010)
    # np.save( target_path + 'dev_audio_emobase2010.npy', dev_audio_emobase2010)
    # np.save( target_path + 'test_audio_emobase2010.npy', test_audio_emobase2010)


    # sequenceN
    seqN_npy = []
    with open('E:/data/SER/IEMOCAP/processed/' + target_path_name + '/FC_MFCC12EDA_sequenceN.txt') as f:
        seqN = [int(x.strip()) for x in f.readlines()]
        seqN_npy = np.asarray(seqN)
    
    train_seqN = extract_data_with_ids(seqN_npy, train_ids)
    dev_seqN = extract_data_with_ids(seqN_npy, dev_ids)
    test_seqN = extract_data_with_ids(seqN_npy, test_ids)
    
    np.save(target_path + 'train_seqN.npy', train_seqN)
    np.save(target_path + 'dev_seqN.npy', dev_seqN)
    np.save(target_path + 'test_seqN.npy', test_seqN)

    # label
    label_npy = []
    with open('E:/data/SER/IEMOCAP/processed/' + target_path_name + '/FC_label.txt') as f :
        label = [ int(x.strip()) for x in f.readlines() ]
        label_npy = np.asarray(label)

    train_label = extract_data_with_ids( label_npy, train_ids  )
    dev_label  = extract_data_with_ids( label_npy, dev_ids  )
    test_label = extract_data_with_ids( label_npy, test_ids  )

    np.save( target_path + 'train_label.npy', train_label)
    np.save( target_path + 'dev_label.npy', dev_label)
    np.save( target_path + 'test_label.npy', test_label)

    # trans
    train_nlp_trans = extract_data_with_ids( np.load( 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/FC_trans.npy' ), train_ids  )
    dev_nlp_trans  = extract_data_with_ids( np.load( 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/FC_trans.npy' ), dev_ids  )
    test_nlp_trans  = extract_data_with_ids( np.load( 'E:/data/SER/IEMOCAP/processed/' + target_path_name + '/FC_trans.npy' ), test_ids  )

    np.save( target_path + 'train_nlp_trans.npy', train_nlp_trans)
    np.save( target_path + 'dev_nlp_trans.npy', dev_nlp_trans)
    np.save( target_path + 'test_nlp_trans.npy', test_nlp_trans)

for target_name in target_name_list:
    generate_train_test(target_name)