import numpy as np

def distortion_measure(x0, x1, r, mu, K):
    N = len(x0)
    J = 0
    for n in range(N):
        for k in range(K):
            J = J + r[n, k] * ((x0[n] - mu[k, 0])**2 + (x1[n] - mu[k, 1])**2)
    return J

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
    R = np.c_[np.ones((N, 1), dtype=int), np.zeros((N, 2), dtype=int)]
    print(distortion_measure(X[:, 0], X[:, 1], R, Mu, K))