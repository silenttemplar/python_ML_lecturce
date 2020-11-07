import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from source.ch6.p3.ch6_3_5 import cee_logistic, dcee_logistic
from source.ch6.p3.ch6_3_3 import show3d_logistic, show_data_3d
from source.ch6.p3.ch6_3_2 import show_data
from source.ch6.p3.ch6_3_4 import show_contour_logistic

# 로지스틱 회귀 모델의 매개 변수 검색
def fit_logistic(w_init, x, t):
    res = minimize(cee_logistic, w_init, args=(x, t),
                   jac=dcee_logistic, method="CG")
    return res.x

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_3_data.npz')

    X = data['X']
    T2 = data['T2']

    X_range0 = [-3, 3]
    X_range1 = [-3, 3]

    W_init = [-1, 0, 0]
    W = fit_logistic(W_init, X, T2)
    cee = cee_logistic(W, X, T2)

    print("w0 = {0:.2f}, w1 = {1:.2f}, w2 = {2:.2f}".format(W[0], W[1], W[2]))
    print("CEE = {0:.2f}".format(cee))

    plt.figure(1, figsize=(7, 3))
    plt.subplots_adjust(wspace=0.5)
    Ax = plt.subplot(1, 2, 1, projection='3d')
    show3d_logistic(Ax, W, X_range=(X_range0, X_range1))
    show_data_3d(Ax, X, T2)
    Ax = plt.subplot(1, 2, 2)
    show_data(X, T2)
    show_contour_logistic(W, X_range=(X_range0, X_range1))
    plt.show()