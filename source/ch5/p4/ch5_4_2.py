import numpy as np
import matplotlib.pyplot as plt
from source.ch5.p3.ch5_3_3 import fit_gauss_func, show_gauss_func, mse_gauss_func

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
    M = [2, 4, 7, 9]

    plt.figure(figsize=(10, 2.5))
    plt.subplots_adjust(wspace=0.3)
    for i in range(len(M)):
        W = fit_gauss_func(X_train, T_train, M[i])
        mse = mse_gauss_func(X_test, T_test, W)
        #print('W={0:s}'.format(str(np.round(W, 1))))
        #print("SD={0:.2f}cm".format(np.sqrt(mse)))

        plt.subplot(1, len(M), i+1)
        show_gauss_func(W, (X_min, X_max))
        plt.plot(X_train, T_train, marker='o', linestyle='None',
                 color='white', markeredgecolor='black', label='training')
        plt.plot(X_test, T_test, marker='o', linestyle='None',
                 color='cornflowerblue', markeredgecolor='black', label='test')
        plt.legend(loc='lower right', fontsize=10, numpoints=1)
        plt.xlim(X_min, X_max)
        plt.ylim(130, 180)
        plt.grid(True)
        plt.title("M={0:d}, SD={1:.1f}".format(M[i], np.sqrt(mse)))
    plt.show()