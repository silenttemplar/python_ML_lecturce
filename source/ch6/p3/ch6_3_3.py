import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def logistic(x0, x1, w):
    y = 1 / (1 + np.exp(-(w[0] * x0 + w[1] * x1 + w[2])))
    return y

def show3d_logistic(ax, w, X_range):
    xn = 50
    x0 = np.linspace(X_range[0][0], X_range[0][1], xn)
    x1 = np.linspace(X_range[1][0], X_range[1][1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    y = logistic(xx0, xx1, w)
    ax.plot_surface(xx0, xx1, y, color='blue', edgecolor='gray',
                    rstride=5, cstride=5, alpha=0.3)


def show_data_3d(ax, x, t):
    c = [[.5, .5, .5], [1, 1, 1]]
    for i in range(2):
        ax.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], 1 - i,
                marker='o', color=c[i], markeredgecolor='black',
                linestyle='none', markersize=5, alpha=0.8)
    ax.view_init(elev=25, azim=-30)

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_3_data.npz')

    X = data['X']
    T2 = data['T2']

    X_range0 = [-3, 3]
    X_range1 = [-3, 3]

    W = [-1, -1, -1]
    #plt.figure(figsize=(7.5, 3))
    Ax = plt.subplot(1, 1, 1, projection='3d')
    show3d_logistic(Ax, W, (X_range0, X_range1))
    show_data_3d(Ax, X, T2)
    plt.show()