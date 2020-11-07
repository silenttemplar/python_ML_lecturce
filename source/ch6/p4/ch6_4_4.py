import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from source.ch6.p4.ch6_4_2 import logistic
from source.ch6.p4.ch6_4_3 import cee_logistic, dcee_logistic
from source.ch6.p3.ch6_3_2 import show_data

# 매개변수 검색
def fit_logistic(w_init, x, t):
    res = minimize(cee_logistic, w_init, args=(x, t),
                   jac=dcee_logistic, method="CG")
    return res.x

def show_contour_logistic(w, X_range):
    xn = 30 # 파라미터의 분할 수
    x0 = np.linspace(X_range[0][0], X_range[0][1], xn)
    x1 = np.linspace(X_range[1][0], X_range[1][1], xn)


    xx0, xx1 = np.meshgrid(x0, x1)
    y = np.zeros((xn, xn, 3))
    for i in range(xn):
        wk = logistic(xx0[:, i], xx1[:, i], w)
        for j in range(3):
            y[:, i, j] = wk[:, j]
    for j in range(3):
        cont = plt.contour(xx0, xx1, y[:, :, j],
                           levels=(0.5, 0.9),
                           colors=['cornflowerblue', 'k'])
        cont.clabel(fmt='%1.1f', fontsize=9)
    plt.grid(True)

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_4_data.npz')

    X = data['X']
    T3 = data['T3']

    X_range0 = [-3, 3]
    X_range1 = [-3, 3]

    W_init = np.zeros((3, 3))
    W = fit_logistic(W_init, X, T3)
    cee = cee_logistic(W, X, T3)

    print(np.round(W.reshape((3, 3)), 2))
    print("CEE = {0:.2f}".format(cee))

    plt.figure(figsize=(3, 3))
    show_data(X, T3)
    show_contour_logistic(W, X_range=(X_range0, X_range1))
    plt.show()