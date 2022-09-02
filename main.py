"""
import csv
import numpy as np
from preprocessing.extract_labels import category
in_file = 'E:/data/SER/IEMOCAP/processed/processed_label.txt'
label = []
with open(in_file, 'r') as f:
    label = f.readlines()

label_id = [x.strip() for x in label]
#print(label_id)

in_file = 'E:/data/SER/IEMOCAP/processed/label.csv'
label = []
with open(in_file, 'r') as f:
    csv_reader = csv.reader(f)
    label = [x for x in csv_reader]

label_cat = [x[1] for x in label]
count = np.zeros(len(category), dtype=np.int)

for cat in label_cat:
    count[category[cat]] = count[category[cat]] + 1
for key in category.keys():
    print(key + '\t:' + str(count[category[key]]))
"""
import numpy as np
data=np.load("E:/data/SER/IEMOCAP/processed/four_category/audio_woZ_set01/test_audio_mfcc.npy")
print(data[:10])
print(data.shape)