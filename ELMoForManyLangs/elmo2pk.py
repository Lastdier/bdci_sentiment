import pandas as pd
import numpy as np
import tqdm
from sklearn.externals import joblib


SEQ_LEN = 100

class ELMo(object):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path, 'r'):
            yield line.strip()

    def __len__(self):
        length = []
        with open(self.path, 'r') as f:
            length = f.readline().split('\t')
        return len(length) - 1

elmo = ELMo('my_elmo.ly-1.txt')
# print(len(elmo))

sample_vector = []
temp = []
for i, line in tqdm.tqdm(enumerate(elmo)):
    line_list = line.split('\t')
    if len(line_list) == 1025 and len(temp) < SEQ_LEN:
        temp.append([float(j) for j in line_list[1:]])
    # print(line)
    elif len(line_list) == 1 and len(temp) > 0:
        while len(temp) < SEQ_LEN:
            temp.append([0.] * 1024)
        sample_vector.append(temp)
        temp = []
        # print(line)

while len(temp) < SEQ_LEN:
    temp.append([0.] * 1024)
sample_vector.append(temp)

sample_vector = np.array(sample_vector, dtype=np.float32)
print(sample_vector.shape)
print(sample_vector.dtype)
joblib.dump(sample_vector, 'sample_vector.pk')
