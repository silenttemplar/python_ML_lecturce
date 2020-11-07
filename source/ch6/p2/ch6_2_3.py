import numpy as np
from source.ch6.p2.ch6_2_2 import logistic

def cee_logistic(w, x, t, X):
    y = logistic(x, w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n] * np.log(y[n]) + (1 - t[n]) * np.log(1 - y[n]))
    cee = cee / X[2]
    return cee

def dcee_logistic(w, x, t, X):
    y = logistic(x, w)
    dcee = np.zeros(2)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n]) * x[n]
        dcee[1] = dcee[1] + (y[n] - t[n])
    dcee = dcee / X[2]
    return dcee


if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_2_data.npz')

    X_min = data['X_min']
    X_max = data['X_max']
    X_n = data['X_n']
    X_col = data['X_col']
    X = data['X']
    T = data['T']

    W = [1, 1]
    print(cee_logistic(W, X, T, (X_max, X_min, X_n)))
    print(dcee_logistic(W, X, T, (X_max, X_min, X_n)))