import readhcvdat0 
import numpy as np
import numpy.matlib
from sklearn.cluster import KMeans


def CalEuclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

def GetDisMatrix(vec_data):
    dis_matrix = np.zeros((len(vec_data), len(vec_data)))
    print('dis_matrix %d*%d' % (len(dis_matrix), len(dis_matrix[0])))
    for i in range(len(vec_data)):
        for j in range(len(vec_data) - i):
            dis_matrix[i,j] = CalEuclidDistance(vec_data[i], vec_data[j], True)
            dis_matrix[j,i] = dis_matrix[i,j]
    return dis_matrix


def GetAdjacentMatrixByKNN(dis_matrix, k, sigma=1.0):
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
            adjacent_matrix[j][i] = adjacent_matrix[i][j] # mutually
    return adjacent_matrix

def GetLaplacianMatrix(adjacent_matrix):
    # compute the Degree Matrix: D=sum(A)
    degree_matrix = np.sum(adjacent_matrix, axis=1)
    # compute the Laplacian Matrix: L=D-A
    laplacian_matrix = np.diag(degree_matrix) - adjacent_matrix
    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrt_degree_matrix = np.diag(1.0 / (degree_matrix ** (0.5)))
    return np.dot(np.dot(sqrt_degree_matrix, laplacian_matrix), sqrt_degree_matrix)

def GetEigenMatrix(laplacian_matrix):
    lam, eigen_vector = np.linalg.eig(laplacian_matrix) # H'shape is n*n
    lam = zip(lam, range(len(lam)))
    lam = sorted(lam, key=lambda x:x[0])
    eigen_matrix = np.vstack([eigen_vector[:,i] for (v, i) in lam[:1000]]).T
    return eigen_matrix

def spKmeans(eigen_matrix):
    sp_kmeans = KMeans(n_clusters=4).fit(eigen_matrix)
    return sp_kmeans.labels_

if __name__ == '__main__' :
    vec_data = np.array(readhcvdat0.ReadHcvDat('hcvdat0.csv'))
    dis_matrix = GetDisMatrix(vec_data)
    print('dis_matrix %d*%d' % (len(dis_matrix), len(dis_matrix[0])))
    adjacent_matrix = GetAdjacentMatrixByKNN(dis_matrix, 20)
    laplacian_matrix = GetLaplacianMatrix(adjacent_matrix)
    eigen_matrix = GetEigenMatrix(laplacian_matrix)
    labels = spKmeans(eigen_matrix)
    print(labels)

