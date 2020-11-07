import numpy as np
from source.ch9.p3.ch9_3_1 import gauss

def mixgauss(x, pi, mu, sigma):
    N, D = x.shape
    K = len(pi)
    p = np.zeros(N)
    for k in range(K):
        p = p + pi[k] * gauss(x, mu[k, :], sigma[k, :, :])
    return p

if __name__ == '__main__':
    x = np.array([[1, 2], [2, 2], [3, 4]])
    pi = np.array([0.3, 0.7])
    mu = np.array([[1, 1], [2, 2]])
    sigma = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 1]]])
    print(mixgauss(x, pi, mu, sigma))