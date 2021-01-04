import readhcvdat0 
import comm
import copy
import numpy as np
import spectral_clustering
from sklearn.cluster import SpectralClustering

if __name__ == '__main__' :
    vec_data, sample_labels = readhcvdat0.ReadHcvDat('hcvdat0.csv')
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
    hcv_dat = spectral_clustering.SpectralClustering(knn_num=5, kmeans_num=5, lam_range=5,
                                                     sigma=1, origin_data=vec_data,
                                                     origin_labels=shuffle_labels)
    labels = hcv_dat.SpectralClustering()
    entropy = comm.CalEntropy(labels, shuffle_labels)
    print(entropy)
    clustering = SpectralClustering(n_clusters=5, assign_labels = "discretize", random_state = 0).fit(vec_data)
    entropy = comm.CalEntropy(clustering.labels_, shuffle_labels)
    print(entropy)
    accuracy = comm.CalAccuracy(clustering.labels_, shuffle_labels)
    print(accuracy)

