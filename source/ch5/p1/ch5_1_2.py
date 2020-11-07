import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# MSE(Mean Squares Error)
def mse_line(x, t, w):
    y = w[0]*x + w[1]
    mse = np.mean((y-t)**2)
    return mse

if __name__ == '__main__':
    #  ch5_1_1.py 저장한 data 로드
    data = np.load('ch5_1_data.npz')
    X = data['X']
    T = data['T']

    # print(X)
    # print(T)

    xn = 100
    w0_range = [-25, 25]
    w1_range = [120, 170]
    x0 = np.linspace(w0_range[0], w0_range[1], xn)
    x1 = np.linspace(w1_range[0], w1_range[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)

    # 평균제곱오차 계산
    J = np.zeros((len(x0), len(x1)))
    for i0 in range(xn):
        for i1 in range(xn):
            J[i1, i0] = mse_line(X, T, (x0[i0], x1[i1]))

    plt.figure(figsize=(9.5, 4))
    plt.subplots_adjust(wspace=0.5)

    ax = plt.subplot(1, 2, 1, projection='3d')
    ax.plot_surface(xx0, xx1, J, rstride=10, cstride=10, alpha=0.3, color='blue', edgecolor='black')
    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([120, 140, 160])
    ax.view_init(20, -60)

    plt.subplot(1, 2, 2)
    cont = plt.contour(xx0, xx1, J, 30, colors='black', levels=[100, 1000, 10000, 100000])
    cont.clabel(fmt='%1.0f', fontsize=8)
    plt.grid(True)
    plt.show()
