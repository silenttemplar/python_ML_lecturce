import numpy as np
import matplotlib.pyplot as plt
from source.ch5.p5.ch5_5_2 import kfold_gause_func

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../p4/ch5_4_data.npz')

    X = data['X']
    T = data['T']
    X_min = data['X_min']
    X_max = data['X_max']
    X_n = data['X_n']

    # 기저함수의 수
    M = range(2, 8)
    # training set, test set 분할 수
    K = 16

    Cv_Gauss_train = np.zeros((K, len(M)))
    Cv_Gauss_test = np.zeros((K, len(M)))
    for i in range(0, len(M)):
        Cv_Gauss_train [:, i], Cv_Gauss_test[:, i] = kfold_gause_func(X, T, M[i], K)

    mean_Gauss_train = np.sqrt(np.mean(Cv_Gauss_train, axis=0))
    mean_Gauss_test = np.sqrt(np.mean(Cv_Gauss_test, axis=0))

    plt.figure(figsize=(4, 3))
    plt.plot(M, mean_Gauss_train,
             marker='o', linestyle='-',
             color='k', markerfacecolor='w',
             label='training')
    plt.plot(M, mean_Gauss_test,
             marker='o', linestyle='-',
             color='cornflowerblue', markeredgecolor='black',
             label='test')
    plt.legend(loc='upper left', fontsize=10)
    plt.ylim(0, 20)
    plt.grid(True)
    plt.show()
