import numpy as np
import matplotlib.pyplot as plt
from source.ch5.p3.ch5_3_3 import fit_gauss_func, mse_gauss_func

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch5_4_data.npz')

    X = data['X']
    T = data['T']
    X_min = data['X_min']
    X_max = data['X_max']
    X_n = data['X_n']

    # training set
    X_train = X[int(X_n / 4 + 1):]
    T_train = T[int(X_n / 4 + 1):]
    # test set
    X_test = X[:int(X_n / 4 + 1)]
    T_test = T[:int(X_n / 4 + 1)]

    # 기저함수
    M = range(2, 10)

    # 측정된 mse
    mse_train = np.zeros(len(M))
    mse_test = np.zeros(len(M))

    # 기저함수 증가에 따른, 평균제곱오차 계산
    for i in range(len(M)):
        W = fit_gauss_func(X_train, T_train, M[i])
        mse_train[i] = np.sqrt(mse_gauss_func(X_train, T_train, W))
        mse_test[i] = np.sqrt(mse_gauss_func(X_test, T_test, W))

    plt.figure(figsize=(5, 4))
    plt.plot(M, mse_train, marker='o', linestyle='-',
             color='black', markerfacecolor='white',
             markeredgecolor='black', label='training')
    plt.plot(M, mse_test, marker='o', linestyle='-',
             color='cornflowerblue', markeredgecolor='black',
             label='test')
    plt.legend(loc='upper left', fontsize=10)
    plt.ylim(0, 12)
    plt.grid(True)
    plt.show()