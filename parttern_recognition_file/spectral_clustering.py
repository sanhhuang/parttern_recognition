import comm
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import math
from sklearn import datasets
from sklearn.cluster import KMeans

class SpectralClustering:
    def __init__(self, knn_num, kmeans_num, lam_range, sigma, origin_data, origin_labels=[]):
        self.__knn_num = knn_num
        self.__kmeans_num = kmeans_num
        self.__lam_range = lam_range
        self.__sigma = sigma
        self.__origin_data = origin_data
        self.__origin_labels = origin_labels
        self.__normalize_data = []
        self.__dis_matrix = []
        self.__adjacent_matrix = []
        self.__laplacian_matrix = []
        self.__eigen_matrix = []
        self.__labels = []
        

    def CalEuclidDistance(self, x1, x2, sqrt_flag=False):
        res = np.sum((x1-x2)**2)
        if sqrt_flag:
            res = np.sqrt(res)
        return res


    def CalCosDistance(self, x1, x2):
        sum_x1 = np.sqrt(np.sum((x1)**2))
        if sum_x1 == 0:
            x1[0] = 0.01
            sum_x1 = 0.01
        sum_x2 = np.sqrt(np.sum((x2) ** 2))
        if sum_x2 == 0:
            x2[0] = 0.01
            sum_x2 = 0.01
        res = np.dot(x1, x2) /(sum_x1 * sum_x2)
        return res


    def GetDisMatrix(self, vec_data):
        dis_matrix = np.zeros((len(vec_data), len(vec_data)))
        for i in range(len(vec_data)):
            for j in range(i+1):
                dis_matrix[i,j] = self.CalEuclidDistance(vec_data[i], vec_data[j])
                # dis_matrix[i, j] = 1 - self.CalCosDistance(vec_data[i], vec_data[j])
                dis_matrix[j,i] = dis_matrix[i,j]
        return dis_matrix


    def GetAdjacentMatrixByKNN(self, dis_matrix, k, sigma=1.0):
        length = len(dis_matrix)
        #定义邻接矩阵
        adjacent_matrix = np.zeros((length,length))
        for i in range(length):
            #对每个样本进行编号
            dist_with_index = zip(dis_matrix[i], range(length))
            #对距离进行排序
            dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
            #取得距离该样本前k个最小距离的编号
            neighbours_id = [dist_with_index[m][1] for m in range(k)] # xi's k nearest neighbours
            #构建邻接矩阵
            for j in neighbours_id: # xj is xi's neighbour
                adjacent_matrix[i][j] = np.exp(-dis_matrix[i][j]/2/sigma/sigma)
                adjacent_matrix[j][i] = adjacent_matrix[i][j]
        return adjacent_matrix


    def GetLaplacianMatrix(seld, adjacent_matrix):
        # compute the Degree Matrix: D=sum(A)
        degree_matrix = np.sum(adjacent_matrix, axis=1)
        # compute the Laplacian Matrix: L=D-A
        laplacian_matrix = np.diag(degree_matrix) - adjacent_matrix
        # laplacian_matrix = np.dot(np.linalg.inv(np.diag(degree_matrix)),laplacian_matrix)
        # return laplacian_matrix
        # normailze
        # D^(-1/2) L D^(-1/2)
        sqrt_degree_matrix = np.diag(1.0 / (degree_matrix ** (0.5)))
        return np.dot(np.dot(sqrt_degree_matrix, laplacian_matrix), sqrt_degree_matrix)


    def GetEigenMatrix(self, laplacian_matrix, lam_range):
        lam, eigen_vector = np.linalg.eig(laplacian_matrix) # H'shape is n*n
        lam = zip(lam, range(len(lam)))
        lam = sorted(lam, key=lambda x:x[0])
        range_lam = int(math.sqrt(len(lam)))
        eigen_matrix = np.vstack([eigen_vector[:,i] for (v, i) in lam[:lam_range]]).T
        # eigen_matrix = np.vstack([eigen_vector[:,i] for (v, i) in lam[-lam_range:]]).T
        return np.real(eigen_matrix)


    def SpKmeans(self, eigen_matrix):
        sp_kmeans = KMeans(n_clusters=self.__kmeans_num).fit(eigen_matrix)
        return sp_kmeans.labels_


    def SpectralClustering(self):
        #boundary, self.__normalize_data = comm.MinMaxNormalize(self.__origin_data)
        self.__normalize_data = self.__origin_data
        self.__dis_matrix = self.GetDisMatrix(self.__normalize_data)
        self.__adjacent_matrix = self.GetAdjacentMatrixByKNN(self.__dis_matrix, self.__knn_num)
        # print(self.__adjacent_matrix)
        self.__laplacian_matrix = self.GetLaplacianMatrix(self.__adjacent_matrix)
        # print(self.__laplacian_matrix)
        self.__eigen_matrix = self.GetEigenMatrix(self.__laplacian_matrix, self.__lam_range)
        self.__labels = self.SpKmeans(self.__eigen_matrix)
        # print(self.__labels)
        # print(self.__origin_labels)
        return self.__labels


if __name__ == '__main__' :
    vec_data, y = datasets.make_circles(n_samples=1000, factor=0.5, noise=0.05)
    shuffle_ix = np.random.permutation(np.arange(len(vec_data)))
    vec_data = vec_data[shuffle_ix]
    y = y[shuffle_ix]
    spectral_clustering = SpectralClustering(20, 2, len(vec_data), 1.0, vec_data, y)
    labels = spectral_clustering.SpectralClustering()
    plt.figure()
    plt.title('spectral cluster result')
    plt.scatter(vec_data[:, 0], vec_data[:, 1], marker='o',c=labels)
    plt.figure()
    plt.title('spectral cluster result first origin')
    plt.scatter(vec_data[:, 0], vec_data[:, 1], marker='o', c=y)
    plt.show()
    entropy = comm.CalEntropy(labels, y)
    print(entropy)
    print(comm.CalAccuracy(labels, y))