import readhcvdat0 
import comm
import copy
import numpy as np
import spectral_clustering

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
    hcv_dat = spectral_clustering.SpectralClustering(10, 5, 5, 1, vec_data, shuffle_labels)
    labels = hcv_dat.SpectralClustering()
    entropy = comm.CalEntropy(labels, shuffle_labels)
    print(entropy)


