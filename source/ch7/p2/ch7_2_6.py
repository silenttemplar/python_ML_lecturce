import numpy as np
import matplotlib.pyplot as plt
from source.ch7.p1.ch7_1_3 import FFNN
from source.ch7.p1.ch7_1_2 import show_data

# 경계선 표시 함수
def show_FFNN(wv, M, K, X_range):
    xn = 60 # 등고선 표시 해상도
    x0 = np.linspace(X_range[0][0], X_range[0][1], xn)
    x1 = np.linspace(X_range[1][0], X_range[1][1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    x = np.c_[np.reshape(xx0, xn * xn, order='C'), np.reshape(xx1, xn * xn, order='C')]
    y, a, z, b = FFNN(wv, M, K, x)

    for ic in range(K):
        f = y[:, ic]
        f = f.reshape(xn, xn)
        f = f.T
        cont = plt.contour(xx0, xx1, f, levels=[0.8, 0.9],
                           colors=['cornflowerblue', 'black'])
        cont.clabel(fmt='%1.1f', fontsize=9)

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch7_2_data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    T_train = data['T_train']
    T_test = data['T_test']

    X_range0 = data['X_range0']
    X_range1 = data['X_range1']

    # 저장한 data 로드
    result = np.load('ch7_2_result.npz')
    M = result['M']
    K = result['K']
    WV = result['WV']
    WV_hist = result['WV_hist']
    Err_train = result['Err_train']
    Err_test = result['Err_test']

    plt.figure(1, figsize=(8, 3.7))
    plt.subplot(1, 2, 1)
    show_data(X_train, T_train)
    show_FFNN(WV, M, K, X_range=(X_range0, X_range1))
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.title('Training Data')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    show_data(X_test, T_test)
    show_FFNN(WV, M, K, X_range=(X_range0, X_range1))
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.title('Test Data')
    plt.grid(True)
    plt.show()