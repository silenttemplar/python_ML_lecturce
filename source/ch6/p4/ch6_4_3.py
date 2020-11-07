import numpy as np
from source.ch6.p4.ch6_4_2 import logistic

# 교차 엔트로피 오차
def cee_logistic(w, x, t):
    X_n = x.shape[0]
    y = logistic(x[:, 0], x[:, 1], w)
    cee = 0
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            cee = cee - (t[n, k] * np.log(y[n, k]))
    cee = cee / X_n
    return cee

# 교차 엔트로피 오차의 미분
def dcee_logistic(w, x, t):
    X_n = x.shape[0]

    y = logistic(x[:, 0], x[:, 1], w)

    # (클래스의 수 K) x (x의 차원 D+1)
    dcee = np.zeros((3, 3))
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            dcee[k, :] = dcee[k, :] - (t[n, k] - y[n, k])* np.r_[x[n, :], 1]
    dcee = dcee / X_n
    return dcee.reshape(-1)

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_4_data.npz')

    X = data['X']
    T3 = data['T3']

    W = np.array(range(1, 10))
    print(cee_logistic(W, X, T3))
