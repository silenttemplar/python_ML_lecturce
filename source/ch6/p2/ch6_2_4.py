import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from source.ch6.p2.ch6_2_3 import cee_logistic

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_2_data.npz')

    X_min = data['X_min']
    X_max = data['X_max']
    X_n = data['X_n']
    X_col = data['X_col']
    X = data['X']
    T = data['T']

    xn = 80
    w_range = np.array([[0, 15], [-15, 0]])
    x0 = np.linspace(w_range[0, 0], w_range[0, 1], xn)
    x1 = np.linspace(w_range[1, 0], w_range[1, 1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    C = np.zeros((len(x1), len(x0)))
    w = np.zeros(2)
    for i0 in range(xn):
        for i1 in range(xn):
            w[0] = x0[i0]
            w[1] = x1[i1]
            C[i0, i1] = cee_logistic(w, X, T, (X_max, X_min, X_n))

    plt.figure(figsize=(12, 5))
    plt.subplots_adjust(wspace=0.5)
    ax = plt.subplot(1, 2, 1, projection='3d')
    ax.plot_surface(xx0, xx1, C, color='blue', edgecolor='black',
                    rstride=10, cstride=10, alpha=0.3)
    ax.set_xlabel('$w_0$', fontsize=14)
    ax.set_ylabel('$w_1$', fontsize=14)
    ax.set_xlim(0, 15)
    ax.set_ylim(-15, 0)
    ax.view_init(30, -95)

    plt.subplot(1, 2, 2)
    cont = plt.contour(xx0, xx1, C, 20, colors='black',
                       levels=[0.26, 0.4, 0.8, 1.6, 3.2, 6.4])
    plt.xlabel('$w_0$', fontsize=14)
    plt.ylabel('$w_1$', fontsize=14)
    plt.grid(True)
    plt.show()
