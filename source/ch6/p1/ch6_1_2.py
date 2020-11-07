import numpy as np
import matplotlib.pyplot as plt

def show_data(x, t, X):
    K = np.max(t) + 1
    for k in range(K):
        plt.plot(x[t == k], t[t == k], X[2][k],
                 alpha=0.5, linestyle='None', marker='o')
        plt.grid(True)
        plt.ylim(-.5, 1.5)
        plt.xlim(X[0], X[1])
        plt.yticks([0, 1])

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_1_data.npz')

    X_min = data['X_min']
    X_max = data['X_max']
    X_n = data['X_n']
    X_col = data['X_col']

    X = data['X']
    T = data['T']

    fig = plt.figure(figsize=(3, 3))
    show_data(X, T, (X_min, X_max, X_col))
    plt.show()