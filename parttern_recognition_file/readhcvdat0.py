import os
import numpy as np

def ReadHcvDat(file_name):
# open file
    #file_name = 'Sales_Transactions_Dataset_Weekly.csv'
    origin_file = open(file_name, 'r')

    # get line in file
    line = origin_file.readline()
    vec = line.split(',' , -1)
    for i in range(len(vec)):
        if vec[i] == '\"ALB\"':
            break

    all_data = []
    labels = []
    dictionary = {} 
    while line:
        line = origin_file.readline()
        vec = line.split(',', -1)
        if len(vec) < i:
            continue
        all_data.append([float(x) if x.find('NA') < 0 else 0 for x in vec[i:]])
        labels.append(vec[1][1:-2])
        if  dictionary.get(labels[-1], -1) == -1:
            dictionary[labels[-1]] = []
        dictionary[labels[-1]].append(all_data[-1])
    print('%s : %d * %d' % (file_name, len(all_data), len(all_data[0])))
    for i in range(len(all_data[0])):
        mean = np.mean(dictionary[labels[i]][:][i])
        for j in range(len(all_data)):
            all_data[j][i] = all_data[j][i] if all_data[j][i] != 0 else mean

    origin_file.close()
    return all_data, labels

if __name__ == '__main__':
    ReadHcvDat('hcvdat0.csv')
