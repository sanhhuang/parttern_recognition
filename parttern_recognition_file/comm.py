import numpy as np
from math import log
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates


def ShowData(data, origin_labels):
    # 处理类标签数据
    target = {}
    cur_target = 1
    target_labels = []
    for i in origin_labels:
        if target.get(i, -1) == -1:
            target[i] = cur_target
            cur_target += 1
        target_labels.append(target[i])
    data_dict = {}
    for feature in range(len(data[0])):
        data_dict[feature] = data[:, feature]
    data_dict["target_labels"] = target_labels
    # 合成dataFrame
    pd_data = pd.DataFrame(data_dict)
    # 画图
    plt.figure()
    parallel_coordinates(pd_data, "target_labels")
    plt.legend(fancybox=True, shadow=True)
    plt.show()


def MinMaxNormalize(data_array):
    array_size = np.shape(data_array)
    print(array_size)
    boundary = np.zeros((array_size[1], 2))
    for col in range(array_size[1]):
        max_num = np.max(data_array[:, col])
        min_num = np.min(data_array[:, col])
        boundary[col][0] = min_num
        boundary[col][1] = max_num
        for row in range(array_size[0]):
            data_array[row][col] = 1 if (max_num - min_num) == 0 else ((data_array[row][col] - min_num) / (max_num - min_num))
    return boundary, data_array


def CalEntropy(labels, origin_labels):
    ij_dictionary = {}
    for i in range(len(labels)):
        if ij_dictionary.get(labels[i], -1) == -1:
            ij_dictionary[labels[i]] = [{}, 0]
        if ij_dictionary[labels[i]][0].get(origin_labels[i], -1) == -1:
            ij_dictionary[labels[i]][0][origin_labels[i]] = 0
        ij_dictionary[labels[i]][0][origin_labels[i]] += 1
        ij_dictionary[labels[i]][1] += 1
    entropy = 0.0
    for i in ij_dictionary:
        prob = 0.0
        entropy_i = 0.0
        for j in ij_dictionary[i][0]:
            prob = 1.0 * ij_dictionary[i][0][j] / ij_dictionary[i][1]
            entropy_i -= prob * log(prob,2)
        prob = 1.0 * ij_dictionary[i][1] / len(labels)
        entropy += entropy_i * prob
    return entropy


def CalAccuracy(labels, origin_labels):
    ij_dictionary = {}
    labels_map = {}
    for i in range(len(labels)):
        if ij_dictionary.get(labels[i], -1) == -1:
            ij_dictionary[labels[i]] = [{}, 0]
            labels_map[labels[i]] = [origin_labels[i], 0]; 
        if ij_dictionary[labels[i]][0].get(origin_labels[i], -1) == -1:
            ij_dictionary[labels[i]][0][origin_labels[i]] = 0
        ij_dictionary[labels[i]][0][origin_labels[i]] += 1
        ij_dictionary[labels[i]][1] += 1
        if labels_map[labels[i]][1] < ij_dictionary[labels[i]][0][origin_labels[i]]:
            labels_map[labels[i]][0] = origin_labels[i]
            labels_map[labels[i]][1] = ij_dictionary[labels[i]][0][origin_labels[i]]
    # print(ij_dictionary)
    accuracy = 0
    for i in labels_map:
        accuracy += labels_map[i][1]
    return accuracy/len(labels)

if __name__ == '__main__':
    data_array = np.random.random((4, 4))
    print(data_array)
    MinMaxNormalize(data_array)
    print(data_array)
    labels = [1, 2, 2, 2, 1, 2, 2]
    labels2 = [2, 2, 1, 2, 1, 2, 2]
    entropy = CalEntropy(labels, labels2)
    print(entropy)
    accauracy = CalAccuracy(labels, labels2)
    print(accauracy)