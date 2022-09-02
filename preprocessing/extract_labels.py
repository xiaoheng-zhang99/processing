import csv
import os
import sys
from util import file_search
def create_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

#create_folder('E:/data/SER/IEMOCAP\Session1')
create_folder('E:/data/SER/IEMOCAP/processed')
out_file = 'E:/data/SER/IEMOCAP/processed/processed_tran.csv'
#os.system('rm '+ out_file)
os.system('del {}'.format(out_file))


def extract_trans(list_in_file, out_file):
    lines = []

    for in_file in list_in_file:
        cnt = 0

        with open(in_file, 'r') as f:
            lines = f.readlines()
            #print(lines)

        with open(out_file, 'a',newline='') as f:
            csv_writer = csv.writer(f)
            lines = sorted(lines)  # sort based on first element

            for line in lines:
                #print(line)
                name = line.split(':')[0].split(' ')[0].strip()
                #print("name",name)
                # unwanted case
                if name[:3] != 'Ses':  # noise transcription such as reply  M: sorry
                    continue
                elif name[-3:-1] == 'XX':  # we don't have matching pair in label
                    continue
                trans = line.split(':')[1].strip()
                #print("trans",trans)

                cnt += 1
                csv_writer.writerow([name, trans])
                #read = csv.reader(f)
                #for line in read:
            #        print(line)

list_files = []





for x in range(2):
    sess_name = 'Session' + str(x+1)
    path = 'E:/data/SER/IEMOCAP/' + sess_name + '/dialog/transcriptions/'
    file_search(path, list_files)
    list_files = sorted(list_files)
    #print (list_files)
    print(sess_name + ", #sum files: " + str(len(list_files)))

extract_trans(list_files, out_file)

out_file_label = 'E:/data/SER/IEMOCAP/processed/label.csv'
#os.system('rm ' + out_file)
os.system('del {}'.format(out_file_label))
list_category = [
                'ang',
                'hap',
                'sad',
                'neu',
                'fru',
                'exc',
                'fea',
                'sur',
                'dis',
                'oth',
                'xxx'
                ]

category = {}
for c_type in list_category:
    if c_type in category:
        category[c_type]+=1
        #print("cat1",category)
    else:
        category[c_type] = len(category)
        #print("cat2", category)


def find_category(lines):
    is_target = True

    id = ''
    c_label = ''
    list_ret = []

    for line in lines:

        if is_target == True:

            try:
                id = line.split('\t')[1].strip()  # extract ID
                #print(id)
                c_label = line.split('\t')[2].strip()  # extract category
                #print(c_label)
                #print(category)
                if not c_label in category:
                    print("ERROR nokey" + c_label)
                    sys.exit()
                list_ret.append([id, c_label])
                #print('here')
                is_target = False

            except:
                print("ERROR " + line)
                sys.exit()

        else:
            if line == '\n':
                is_target = True

    return list_ret


def extract_labels(list_in_file, out_file_label):
    id = ''
    lines = []
    list_ret = []

    for in_file in list_in_file:
        with open(in_file, 'r') as f:
            lines = f.readlines()
            #print(lines)
            lines = lines[2:]  # remove head
            list_ret = find_category(lines)
            #print(list_ret)

        list_ret = sorted(list_ret)  # sort based on first element

        with open(out_file_label, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(list_ret)
        # [schema] ID, label [csv]


list_files = []
list_avoid_dir = ['Attribute', 'Categorical', 'Self-evaluation']

for x in range(2):
    sess_name = 'Session' + str(x + 1)

    path = 'E:/data/SER/IEMOCAP/' + sess_name + '/dialog/EmoEvaluation/'
    file_search(path, list_files, list_avoid_dir)
    list_files = sorted(list_files)

    print(sess_name + ", #sum files: " + str(len(list_files)))
#print(list_files)
extract_labels(list_files, out_file_label)

lines = []
with open('E:/data/SER/IEMOCAP/processed/label.csv') as f:
    csv_reader = csv.reader(f)
    lines = [x for x in csv_reader]
with open('E:/data/SER/IEMOCAP/processed/processed_label.txt', 'w') as f:
    with open('E:/data/SER/IEMOCAP/processed/processed_ids.txt', 'w') as f2:

        for line in lines:
            #print(line)
            if line[1] == 'ang':
                f.write('0\n')
                f2.write(line[0] + '\n')
            elif line[1] == 'hap':
                f.write('1\n')
                f2.write(line[0] + '\n')
            elif line[1] == 'exc':
                f.write('1\n')
                f2.write(line[0] + '\n')
            elif line[1] == 'sad':
                f.write('2\n')
                f2.write(line[0] + '\n')
            elif line[1] == 'neu':
                f.write('3\n')
                f2.write(line[0] + '\n')
            else:
                f.write('-1\n')
lines = []
with open('E:/data/SER/IEMOCAP/processed/processed_label.txt') as f:
    lines = f.readlines()
lines = [x.strip() for x in lines]

print(len([x for x in lines if x=='0']))

#print(len([x for x in lines if x == '0']))
