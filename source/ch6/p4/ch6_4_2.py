import numpy as np

def logistic(x0, x1, w):
    K = 3

    w = w.reshape((3, 3))
    n = len(x1)
    y = np.zeros((n, K))

    for k in range(K):
        y[:, k] = np.exp(w[k, 0] * x0 + w[k, 1] * x1 + w[k , 2])

    wk = np.sum(y, axis=1)
    wk = y.T / wk
    y = wk.T
    return y


if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_4_data.npz')

    X = data['X']
    #T3 = data['T3']

    W = np.array(range(1, 10))
    #print(W)
    y = logistic(X[:3, 0], X[:3, 1], W)
    print(np.round(y, 3))