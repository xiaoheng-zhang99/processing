import numpy as np
import codecs
import pickle
import operator


def loadGloveModel(gloveFile):
    print
    "Loading Glove Model"
    f = codecs.open(gloveFile, 'rb')

    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model

    f.close()


def cal_coverage(voca, glove):
    cnt = 0
    for token in voca.keys():
        if token in glove:
            continue;
        else:
            cnt = cnt + 1

    print( '# missing token : ' + str(cnt))
    print( 'coverage : ' + str(1 - (cnt / float(len(voca)))))


def create_glove_embedding(voca, glove):
    # sorting voca dic by value
    sorted_voca = sorted(voca.items(), key=operator.itemgetter(1))

    list_glove_voca = []

    cnt = 0

    for token, value in sorted_voca:
        if token in glove:
            list_glove_voca.append(glove[token])
        else:
            if token == '_PAD_':
                print('add PAD as 0s')
                list_glove_voca.append(np.zeros(300))
            else:
                list_glove_voca.append(np.random.uniform(-0.25, 0.25, 300).tolist())
                cnt = cnt + 1
    print('coverage : ' + str(1 - (cnt / float(len(voca)))))
    return list_glove_voca


glove =loadGloveModel(r'E:\data\NLP\glove.840B.300d.txt')

# glove_twit = loadGloveModel('../data/raw/embedding/glove.twitter.27B.200d.txt')

dic = {}
with open('E:/data/SER/IEMOCAP/processed/dic.pkl','rb') as f:
    dic = pickle.load(f)
print('total dic size : ' + str(len(dic)))

cal_coverage(dic, glove)
# cal_coverage(dic, glove_twit)
list_glove_voca = create_glove_embedding(dic, glove)
print(len(list_glove_voca))

np_glove = np.asarray(list_glove_voca, dtype=np.float32)
print(np.shape(np_glove))
np.save('E:/data/SER/IEMOCAP/processed/W_embedding.npy', np_glove)