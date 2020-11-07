import numpy as np
import matplotlib.pyplot as plt
from source.ch9.p1.ch9_1_4 import step2_kmeans
from source.ch9.p1.ch9_1_3 import step1_kmeans
from source.ch9.p2.ch9_2_1 import distortion_measure

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../ch9_1_data.npz')

    #N = data['N']
    K = data['K']
    X_range0 = data['X_range0']
    X_range1 = data['X_range1']
    X = data['X']
    X_col = data['X_col']

    N = X.shape[0]
    Mu = np.array([[-2, 1], [-2, 0], [-2, -1]])
    R = np.c_[np.zeros((N, 1), dtype=int), np.zeros((N, 2), dtype=int)]

    max_it = 10
    it = 0
    DM = np.zeros(max_it)
    for it in range(max_it):
        R = step1_kmeans(X[:, 0], X[:, 1], Mu, N, K)
        DM[it] = distortion_measure(X[:, 0], X[:, 1], R, Mu, K)
        Mu = step2_kmeans(X[:, 0], X[:, 1], R, N, K)
    print(np.round(DM, 2))

    plt.figure(2, figsize=(4, 4))
    plt.plot(DM, color='black', linestyle='-', marker='o')
    plt.ylim(40, 80)
    plt.grid(True)
    plt.show()