import read_Data_User_Modeling_Dataset_Hamdi
import numpy as np
import spectral_clustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import math
import comm
import copy

if __name__ == '__main__':
    vec_data, sample_labels = read_Data_User_Modeling_Dataset_Hamdi.ReadUserModelingDataSetHamdi(
        'Data_User_Modeling_Dataset_Hamdi.xls')
    vec_data = np.array(vec_data)
    boundary, vec_data = comm.MinMaxNormalize(vec_data)
    shuffle_ix = np.random.permutation(np.arange(len(vec_data)))
    vec_data = vec_data[shuffle_ix]
    shuffle_labels = copy.copy(sample_labels)
    index = 0
    for i in shuffle_ix:
        shuffle_labels[index] = sample_labels[i]
        index += 1
    comm.ShowData(vec_data, shuffle_labels)
    seeds_dataset = spectral_clustering.SpectralClustering(knn_num=10, kmeans_num=4, lam_range=5,
                                                           sigma=1.5, origin_data=vec_data,
                                                           origin_labels=shuffle_labels)
    labels = seeds_dataset.SpectralClustering()
    entropy = comm.CalEntropy(labels, shuffle_labels)
    print(entropy)
    clustering = SpectralClustering(n_clusters=4, assign_labels = "discretize", random_state = 0).fit(vec_data)
    entropy = comm.CalEntropy(clustering.labels_, shuffle_labels)
    print(entropy)
