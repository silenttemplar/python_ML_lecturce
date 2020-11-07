import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from source.ch6.p2.ch6_2_3 import cee_logistic, dcee_logistic
from source.ch6.p2.ch6_2_2 import show_logistic
from source.ch6.p1.ch6_1_2 import show_data

def fit_logistic(w_init, x, t, X):
    res1 = minimize(cee_logistic, w_init, args=(x, t, X), jac=dcee_logistic, method='CG')
    return res1.x

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_2_data.npz')

    X_min = data['X_min']
    X_max = data['X_max']
    X_n = data['X_n']
    X_col = data['X_col']
    X = data['X']
    T = data['T']

    W_init = [1, -1]
    W = fit_logistic(W_init, X, T, (X_min, X_max, X_n))
    print("w0 = {0:.2f}, w1 = {1:.2f}".format(W[0], W[1]))

    cee = cee_logistic(W, X, T, (X_min, X_max, X_n))
    print('CEE = {0:.2f}'.format(cee))

    plt.figure(1, figsize=(3, 3))
    B = show_logistic(W, (X_min, X_max))
    print("Boundary = {0:.2f} g".format(B))
    show_data(X, T, (X_min, X_max, X_col))
    plt.ylim(-.5, 1.5)
    plt.xlim(X_min, X_max)
    plt.show()
