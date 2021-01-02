import numpy as np
import copy
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class SpectralCluster:
    def __init__(self, origin_data):
        self.__origin_data = origin_data


    def EuclidDistance(self, x1, x2, sqrt_flag=False):
        res = np.sum((x1-x2)**2)
        if sqrt_flag:
            res = np.sqrt(res)
        return res


    def CalDistanceMatrix(self, X):
        X = np.array(X)
        S = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                S[i][j] = 1.0 * self.EuclidDistance(X[i], X[j])
                S[j][i] = S[i][j]
        return S


    def GetAdjacentMatrixByKNN(self, S, k, sigma=1.0):
        N = len(S)
        #定义邻接矩阵
        A = np.zeros((N,N))
        for i in range(N):
        #对每个样本进行编号
            dist_with_index = zip(S[i], range(N))
        #对距离进行排序
            dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        #取得距离该样本前k个最小距离的编号
            neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours
        #构建邻接矩阵
            for j in neighbours_id: # xj is xi's neighbour
                A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
                A[j][i] = A[i][j] # mutually
        return A


    def CalLaplacianMatrix(self, adjacentMatrix):

        # compute the Degree Matrix: D=sum(A)
        degreeMatrix = np.sum(adjacentMatrix, axis=1)

        # compute the Laplacian Matrix: L=D-A
        laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

        # normailze
        # D^(-1/2) L D^(-1/2)
        sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
        return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

    def GetHMatrix(self, L_sys):
        lam, V = np.linalg.eig(L_sys)  # H'shape is n*n
        lam = zip(lam, range(len(lam)))
        lam = sorted(lam, key=lambda x: x[0])
        H = np.vstack([V[:, i] for (v, i) in lam]).T
        H = np.asarray(H).astype(float)
        return H


    def SpKmeans(self, H, num_clusters):
        sp_kmeans = KMeans(n_clusters=num_clusters).fit(H)
        return sp_kmeans.labels_


    def Cluster(self):
        return self.SpKmeans(self.GetHMatrix(self.CalLaplacianMatrix(self.GetAdjacentMatrixByKNN(self.CalDistanceMatrix(self.__origin_data), 20))), 2)


if __name__=='__main__':
    x1, y1 = datasets.make_circles(n_samples=1000, factor=0.5, noise=0.05)
    shuffle_ix = np.random.permutation(np.arange(len(x1)))
    x1 = x1[shuffle_ix]
    y1 = y1[shuffle_ix]
    cluster = SpectralCluster(x1)
    labels = cluster.Cluster()
    print(labels)
    plt.figure()
    plt.title('spectral cluster result first')
    plt.scatter(x1[:, 0], x1[:, 1], marker='o',c=labels)
    plt.figure()
    plt.title('spectral cluster result first origin')
    plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)
    plt.show()
