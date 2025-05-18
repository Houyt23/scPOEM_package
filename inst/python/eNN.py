import numpy as np
from scipy import sparse
import math
from scipy.spatial import distance
#import argparse


def get_threshold(E_g):
    q_FDR = 0.05

    num = math.ceil(q_FDR * E_g.shape[0] * (E_g.shape[0]-1)/2)
    E_g_dense = E_g.todense() if sparse.issparse(E_g) else E_g
    
    temp = distance.cdist(E_g_dense, E_g_dense, 'euclidean')

    rows, cols = temp.shape
    m, n = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")  # 生成行、列索引

    mask = m < n
    filtered_m = m[mask].flatten()  # i + m，转换为一维
    filtered_n = n[mask].flatten()        # 保持 n 的索引
    temp_values = temp[mask].flatten()    # 保持对应的距离值

    result = np.column_stack((temp_values, filtered_m, filtered_n))
    result = result[result[:, 0].argsort()]
    result = result[:num, :]  # 只保留最小的 num 个距离

    rows = result[:,1].astype(int)
    cols = result[:,2].astype(int)
    vals = np.exp(-result[:,0])

    sparse_matrix = sparse.coo_matrix((vals, (rows, cols)), shape=(E_g.shape[0], E_g.shape[0]))
    sparse_matrix = sparse_matrix + sparse_matrix.T
    return sparse_matrix


def get_eNN(E_g):
    W_eNN = get_threshold(E_g)
    return W_eNN



#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--dirpath", type=str, default="data_example/compare/")
#    parser.add_argument("--state1", type=str, default="S1")
#    parser.add_argument("--state2", type=str, default="S2")
#    args = parser.parse_args()
#    get_eNN(os.path.join(args.dirpath, args.state1))
#    get_eNN(os.path.join(args.dirpath, args.state2))

    