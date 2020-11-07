import numpy as np
import matplotlib.pyplot as plt
from source.ch9.p1.ch9_1_3 import step1_kmeans
from source.ch9.p1.ch9_1_2 import show_prm

def step2_kmeans(x0, x1, r, N, K):
    mu = np.zeros((K, 2))
    for k in range(K):
        mu[k, 0] = np.sum(r[:, k] * x0) / np.sum(r[:, k])
        mu[k, 1] = np.sum(r[:, k] * x1) / np.sum(r[:, k])
    return mu

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
    
    R = step1_kmeans(X[:, 0], X[:, 1], Mu, N, K)
    Mu = step2_kmeans(X[:, 0], X[:, 1], R, N, K)

    plt.figure(figsize=(4, 4))
    show_prm(X, R, Mu, N, K, X_col)
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.title('Step 1')
    plt.show()