import numpy as np
import matplotlib.pyplot as plt
from source.ch9.p1.ch9_1_1 import show_data

def show_prm(x, r, mu, N, K, col):
    for k in range(K):
        plt.plot(x[r[:, k] == 1, 0], x[r[:, k] == 1, 1],
                 marker='o', markerfacecolor=col[k],
                 markeredgecolor='k',
                 markersize=6, alpha=0.5, linestyle='none')
        plt.plot(mu[k, 0], mu[k, 1],
                 marker='*', markerfacecolor=col[k],
                 markeredgecolor='k', markeredgewidth=1,
                 markersize=15)
    plt.grid(True)

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../ch9_1_data.npz')

    N = data['N']
    K = data['K']
    X_range0 = data['X_range0']
    X_range1 = data['X_range1']
    X = data['X']
    X_col = data['X_col']

    Mu = np.array([[-2, 1], [-2, 0], [-2, -1]])
    R = np.c_[np.zeros((N, 1), dtype=int), np.zeros((N, 2), dtype=int)]

    plt.figure(figsize=(4, 4))
    show_data(X)
    show_prm(X, R, Mu, N, K, X_col)
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.title('initial Mu and R')
    plt.show()