import numpy as np
import matplotlib.pyplot as plt
from source.ch6.p3.ch6_3_3 import logistic

# 크로스 엔트로피 오차
def cee_logistic(w, x, t):
    X_n = x.shape[0]
    y = logistic(x[:, 0], x[:, 1], w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n, 0] * np.log(y[n]) +
                     (1 - t[n, 0]) * np.log(1 - y[n]))
    cee = cee / X_n
    return cee

# 크로스 엔트로피 오차의 미분
def dcee_logistic(w, x, t):
    X_n=x.shape[0]
    y = logistic(x[:, 0], x[:, 1], w)
    dcee = np.zeros(3)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n, 0]) * x[n, 0]
        dcee[1] = dcee[1] + (y[n] - t[n, 0]) * x[n, 1]
        dcee[2] = dcee[2] + (y[n] - t[n, 0])
    dcee = dcee / X_n
    return dcee

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_3_data.npz')

    X = data['X']
    T2 = data['T2']

    X_range0 = [-3, 3]
    X_range1 = [-3, 3]

    W = [-1, -1, -1]

    print(cee_logistic(W, X, T2))
    print(dcee_logistic(W, X, T2))