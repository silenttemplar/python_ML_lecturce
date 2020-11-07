import numpy as np
import matplotlib.pyplot as plt
from source.ch9.p1.ch9_1_4 import step2_kmeans
from source.ch9.p1.ch9_1_3 import step1_kmeans
from source.ch9.p1.ch9_1_2 import show_prm

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../ch9_1_data.npz')

    N = data['N']
    K = data['K']
    X_range0 = data['X_range0']
    X_range1 = data['X_range1']
    X = data['X']
    X_col = data['X_col']

    # 초기중심
    Mu = np.array([[-2, 1], [-2, 0], [-2, -1]])
    # 반복횟수
    max_it = 6

    plt.figure(1, figsize=(10, 6.5))
    for it in range(0, max_it):
        R = step1_kmeans(X[:, 0], X[:, 1], Mu, N, K)

        plt.subplot(2, 3, it+1)
        show_prm(X, R, Mu, N, K, X_col)
        plt.xticks(range(X_range0[0], X_range0[1]), "")
        plt.yticks(range(X_range1[0], X_range1[1]), "")
        plt.title('{0:d}'.format(it + 1))

        Mu = step2_kmeans(X[:, 0], X[:, 1], R, N, K)
    plt.show()