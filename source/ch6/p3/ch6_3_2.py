import numpy as np
import matplotlib.pyplot as plt

def show_data(x, t):
    wk, K = t.shape
    c = [[.5, .5, .5], [1, 1, 1], [0, 0, 0]]
    for k in range(K):
        plt.plot(x[t[:, k] == 1, 0], x[t[:, k] == 1, 1],
                 linestyle='none', markeredgecolor='black',
                 marker='o', color=c[k], alpha=0.8)
        plt.grid(True)

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_3_data.npz')

    X = data['X']
    T2 = data['T2']

    X_range0 = [-3, 3]
    X_range1 = [-3, 3]

    plt.figure(figsize=(7.5, 3))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1, 2, 1)
    show_data(X, T2)
    plt.xlim(X_range0)
    plt.ylim(X_range1)

    plt.subplot(1, 2, 2)
    show_data(X, T3)
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.show()