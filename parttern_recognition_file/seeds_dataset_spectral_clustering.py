import readseeds_dataset 
import numpy as np
import spectral_clustering
import comm
import copy


if __name__ == '__main__' :
    vec_data, sample_labels = readseeds_dataset.ReadSeedsDataSet('seeds_dataset.txt')
    vec_data = np.array(vec_data)
    boundary, vec_data = comm.MinMaxNormalize(vec_data)
    shuffle_ix = np.random.permutation(np.arange(len(vec_data)))
    vec_data = vec_data[shuffle_ix]
    shuffle_labels = copy.copy(sample_labels)
    index = 0
    for i in shuffle_ix:
        shuffle_labels[index] = sample_labels[i]
        index += 1
    seeds_dataset = spectral_clustering.SpectralClustering(10, 4, 10, 1.0, vec_data, shuffle_labels)
    labels = seeds_dataset.SpectralClustering()
    entropy = comm.CalEntropy(labels, shuffle_labels)
    print(entropy)

