# coding:utf-8
import os
import pandas as pd
import numpy as np

Datasets = ['AIDS', 'FRANKENSTEIN', 'Mutagenicity', 'NCI1', 'NCI109', 'DD', 'PROTEINS',
            'COIL-DEL', 'COIL-RAG', 'Letter-high', 'Letter-low', 'Letter-med', 'COLLAB',
            'IMDB-BINARY', 'IMDB-MULTI', 'COLORS-3']

Method = ['MatPool']

res = []
for dataset in Datasets:
    for file in os.listdir():
        temp_dataset = file.split('_')[0]
        if dataset == temp_dataset:
            results = pd.read_csv(file + '/' + 'results.csv', header=None)[0]
            res0 = np.array([float(temp) for temp in results[0].split(' ')]).reshape(1, -1)
            res1 = np.array([float(temp) for temp in results[1].split(' ')]).reshape(1, -1)
            result = np.concatenate((res0, res1), axis=0)
            idx = np.argmax(result[:, -4])
            res.append(result[idx, -1]) # -2 acc -1 std

res = np.array(res)*100
a = 1

