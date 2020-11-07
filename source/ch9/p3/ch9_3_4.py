import numpy as np
import matplotlib.pyplot as plt
from source.ch9.p3.ch9_3_3 import show_contour_mixgauss

def show_mixgauss_prm(K, x, gamma, pi, mu, sigma, x_range, x_col):
    N, D = x.shape
    show_contour_mixgauss(pi, mu, sigma, x_range)
    for n in range(N):
        col = gamma[n, 0]*x_col[0] + gamma[n, 1]*x_col[1] + gamma[n, 2]*x_col[2]
        plt.plot(x[n, 0], x[n, 1],
                 marker='o', color=tuple(col),
                 markeredgecolor='black', markersize=6,
                 alpha=0.5)
    for k in range(K):
        plt.plot(mu[k, 0], mu[k, 1],
                 marker='*', markerfacecolor=tuple(x_col[k]),
                 markersize=15, markeredgecolor='k', markeredgewidth=1)
    plt.grid(True)

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../ch9_1_data.npz')

    #N = data['N']
    #K = data['K']
    X_range0 = data['X_range0']
    X_range1 = data['X_range1']
    X = data['X']
    #X_col = data['X_col']

    N = X.shape[0]
    K = 3

    Pi = np.array([0.33, 0.33, 0.34])
    Mu = np.array([[-2, 1], [-2, 0], [-2, -1]])
    Sigma = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    Gamma = np.c_[np.ones((N, 1)), np.zeros((N, 2))]

    X_col = np.array([[0.4, 0.6, 0.95], [1, 1, 1], [0, 0, 0]])

    plt.figure(1, figsize=(4, 4))
    show_mixgauss_prm(K, X, Gamma, Pi, Mu, Sigma, (X_range0, X_range1), X_col)
    plt.show()


