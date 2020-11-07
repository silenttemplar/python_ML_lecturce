import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from source.ch9.p3.ch9_3_2 import mixgauss

def show_contour_mixgauss(pi, mu, sigma, X_range):
    xn = 40
    X_range0 = X_range[0]
    X_range1 = X_range[1]

    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)

    x = np.c_[np.reshape(xx0, xn * xn), np.reshape(xx1, xn * xn)]
    f = mixgauss(x, pi, mu, sigma)
    f = f.reshape(xn, xn)
    f = f.T
    plt.contour(x0, x1, f, 10, colors='gray')

def show3d_mixgauss(ax, pi, mu, sigma, X_range):
    xn = 40
    x0 = np.linspace(X_range[0][0], X_range[0][1], xn)
    x1 = np.linspace(X_range[1][0], X_range[1][1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    x = np.c_[np.reshape(xx0, xn * xn), np.reshape(xx1, xn * xn)]
    f = mixgauss(x, pi, mu, sigma)
    f = f.reshape(xn, xn)
    f = f.T
    ax.plot_surface(xx0, xx1, f,
                    rstride=2, cstride=2, alpha=0.3,
                    color='blue', edgecolor='black')

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../ch9_1_data.npz')

    X_range0 = data['X_range0']
    X_range1 = data['X_range1']

    pi = np.array([0.2, 0.4, 0.4])
    mu = np.array([[-2, -2], [-1, 1], [1.5, 1]])
    sigma = np.array([[[.5, 0], [0, .5]], [[1, 0.25], [0.25, .5]], [[.5, 0], [0, .5]]])

    Fig = plt.figure(1, figsize=(8, 3.5))
    Fig.add_subplot(1, 2, 1)
    show_contour_mixgauss(pi, mu, sigma, (X_range0, X_range1))
    plt.grid(True)

    Ax = Fig.add_subplot(1, 2, 2, projection='3d')
    show3d_mixgauss(Ax, pi, mu, sigma, (X_range0, X_range1))
    Ax.set_zticks([0.05, 0.10])
    Ax.set_xlabel('$x_0$', fontsize=14)
    Ax.set_ylabel('$x_1$', fontsize=14)
    Ax.view_init(40, -100)
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.show()