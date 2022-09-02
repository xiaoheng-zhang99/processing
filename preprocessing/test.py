from util import *
list_files = []
for x in range(5):
    sess_name = 'Session' + str(x + 1)
    path = 'E:/data/SER/IEMOCAP/' + sess_name + '/sentences/wav/'
    file_search(path, list_files)
    list_files = sorted(list_files)
    cnt = 0
    # print("num",len(list_in_file))
    for in_file in list_files:
        if(in_file[-3:]!="wav"):
            print(in_file)
