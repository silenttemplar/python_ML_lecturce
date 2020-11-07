import numpy as np
import matplotlib.pyplot as plt
from source.ch9.p3.ch9_3_1 import gauss
from source.ch9.p3.ch9_3_5 import e_step_mixgauss, m_step_mixgauss

def nlh_mixgauss(x, pi, mu, sigma):
    # x: NxD
    # pi: Kx1
    # mu: KxD
    # sigma: KxDxD
    # output lh: NxK
    N, D = x.shape
    K = len(pi)
    y = np.zeros((N, K))
    for k in range(K):
        y[:, k] = gauss(x, mu[k, :], sigma[k, :, :]) # KxN
    lh = 0
    for n in range(N):
        wk = 0
        for k in range(K):
            wk = wk + pi[k] * y[n, k]
        lh = lh + np.log(wk)
    return -lh

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
    it = 0
    Err = np.zeros(max_it)  # distortion measure

    for it in range(0, max_it):
        Gamma = e_step_mixgauss(X, Pi, Mu, Sigma)
        Err[it] = nlh_mixgauss(X, Pi, Mu, Sigma)
        Pi, Mu, Sigma = m_step_mixgauss(X, Gamma)

    #print(np.round(Err, 2))
    plt.figure(2, figsize=(4, 4))
    plt.plot(np.arange(max_it) + 1,
             Err, color='k', linestyle='-', marker='o')
    # plt.ylim([40, 80])
    plt.grid(True)
    plt.show()