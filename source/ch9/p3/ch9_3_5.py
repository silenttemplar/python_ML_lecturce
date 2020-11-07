import numpy as np
import matplotlib.pyplot as plt
from source.ch9.p3.ch9_3_1 import gauss
from source.ch9.p3.ch9_3_4 import show_mixgauss_prm

def e_step_mixgauss(x, pi, mu, sigma):
    N, D = x.shape
    K = len(pi)
    y = np.zeros((N, K))
    gamma = np.zeros((N, K))

    for k in range(K):
        y[:, k] = gauss(x, mu[k, :], sigma[k, :, :])

    for n in range(N):
        wk = np.zeros(K)
        for k in range(K):
            wk[k] = pi[k] * y[n, k]
        gamma[n, :] = wk / np.sum(wk)
    return gamma

def m_step_mixgauss(x, gamma):
    N, D = x.shape
    N, K = gamma.shape

    # pi를 계산
    pi = np.sum(gamma, axis=0) / N

    # mu을 계산
    mu = np.zeros((K, D))
    for k in range(K):
        for d in range(D):
            mu[k, d] = np.dot(gamma[:, k], x[:, d]) / np.sum(gamma[:, k])

    # sigma를 계산
    sigma = np.zeros((K, D, D))
    for k in range(K):
        for n in range(N):
            wk = x - mu[k, :]
            wk = wk[n, :, np.newaxis]
            sigma[k, :, :] = sigma[k, :, :] + gamma[n, k] * np.dot(wk, wk.T)
        sigma[k, :, :] = sigma[k, :, :] / np.sum(gamma[:, k])
    return pi, mu, sigma

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../ch9_1_data.npz')

    X_range0 = data['X_range0']
    X_range1 = data['X_range1']

    X = data['X']
    K = 3

    Pi = np.array([0.33, 0.33, 0.34])
    Mu = np.array([[-2, 1], [-2, 0], [-2, -1]])
    Sigma = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    X_col = np.array([[0.4, 0.6, 0.95], [1, 1, 1], [0, 0, 0]])

    Gamma = e_step_mixgauss(X, Pi, Mu, Sigma)
    Pi, Mu, Sigma = m_step_mixgauss(X, Gamma)

    plt.figure(1, figsize=(4, 4))
    show_mixgauss_prm(K, X, Gamma, Pi, Mu, Sigma, (X_range0, X_range1), X_col)
    plt.show()