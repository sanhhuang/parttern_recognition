import readseeds_dataset 
import numpy as np
import spectral_clustering
import comm
import copy
from sklearn.cluster import SpectralClustering

if __name__ == '__main__' :
    vec_data, sample_labels = readseeds_dataset.ReadSeedsDataSet('seeds_dataset.txt')
    vec_data = np.array(vec_data)
    boundary, vec_data = comm.MinMaxNormalize(vec_data)
    shuffle_ix = np.random.permutation(np.arange(len(vec_data)))
    shuffle_labels = copy.copy(sample_labels)
    index = 0
    for i in shuffle_ix:
        shuffle_labels[index] = sample_labels[i]
        index += 1
    vec_data = vec_data[shuffle_ix]
    comm.ShowData(vec_data, shuffle_labels)
    seeds_dataset = spectral_clustering.SpectralClustering(knn_num=10, kmeans_num=3, lam_range=2,
                                                           sigma=1.0, origin_data=vec_data,
                                                           origin_labels=shuffle_labels)
    labels = seeds_dataset.SpectralClustering()
    entropy = comm.CalEntropy(labels, shuffle_labels)
    print(entropy)
    clustering = SpectralClustering(n_clusters=3, assign_labels = "discretize", random_state = 0).fit(vec_data)
    entropy = comm.CalEntropy(clustering.labels_, shuffle_labels)
    print(entropy)
    accuracy = comm.CalAccuracy(clustering.labels_, shuffle_labels)
    print(accuracy)

