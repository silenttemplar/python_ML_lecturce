import numpy as np
import matplotlib.pyplot as plt
from source.ch9.p3.ch9_3_5 import e_step_mixgauss, m_step_mixgauss
from source.ch9.p3.ch9_3_4 import show_mixgauss_prm

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../ch9_1_data.npz')

    X_range0 = data['X_range0']
    X_range1 = data['X_range1']

    N = data['N']
    X = data['X']
    K = 3

    Pi = np.array([0.3, 0.3, 0.4])
    Mu = np.array([[2, 2], [-2, 0], [2, -2]])
    Sigma = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    Gamma = np.c_[np.ones((N, 1)), np.zeros((N, 2))]
    X_col = np.array([[0.4, 0.6, 0.95], [1, 1, 1], [0, 0, 0]])

    max_it = 20  # 반복 횟수

    plt.figure(1, figsize=(10, 6.5))
    i_subplot=1;
    for it in range(0, max_it):
        Gamma = e_step_mixgauss(X, Pi, Mu, Sigma)
        if it<4 or it>17:
            plt.subplot(2, 3, i_subplot)
            show_mixgauss_prm(K, X, Gamma, Pi, Mu, Sigma, (X_range0, X_range1), X_col)
            plt.title("{0:d}".format(it + 1))
            plt.xticks(range(X_range0[0], X_range0[1]), "")
            plt.yticks(range(X_range1[0], X_range1[1]), "")
            i_subplot=i_subplot+1
        Pi, Mu, Sigma = m_step_mixgauss(X, Gamma)
    plt.show()