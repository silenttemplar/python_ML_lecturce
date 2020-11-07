import numpy as np
from source.ch5.p3.ch5_3_3 import fit_gauss_func, mse_gauss_func

# k-fold cross validation
def kfold_gause_func(x, t, m, k):
    n = x.shape[0]
    mse_train = np.zeros(k)
    mse_test = np.zeros(k)

    for i in range(0, k):
        # training set
        x_train = x[np.fmod(range(n), k) != i]
        t_train = t[np.fmod(range(n), k) != i]
        # test set
        x_test = x[np.fmod(range(n), k) == i]
        t_test = t[np.fmod(range(n), k) == i]
        wm = fit_gauss_func(x_train, t_train, m)
        mse_train[i] = mse_gauss_func(x_train, t_train, wm)
        mse_test[i] = mse_gauss_func(x_test, t_test, wm)

    return mse_train, mse_test


if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../p4/ch5_4_data.npz')

    X = data['X']
    T = data['T']
    X_min = data['X_min']
    X_max = data['X_max']
    X_n = data['X_n']

    # 기저함수의 수
    M = 4
    # training set, test set 분할 수
    K = 4

    print(kfold_gause_func(X, T, M, K))