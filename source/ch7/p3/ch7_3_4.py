import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from source.ch7.p1.ch7_1_3 import FFNN

def show_activation3d(ax, v, xn, xx, v_ticks, title_str):
    f = v.copy()
    f = f.reshape(xn, xn)
    f = f.T
    ax.plot_surface(xx[0], xx[1], f, color='blue', edgecolor='black',
                    rstride=1, cstride=1, alpha=0.5)
    ax.view_init(70, -110)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticks(v_ticks)
    ax.set_title(title_str, fontsize=18)

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../p2/ch7_2_data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    T_train = data['T_train']
    T_test = data['T_test']
    X_range0 = data['X_range0']
    X_range1 = data['X_range1']

    # 계산결과
    result = np.load('ch7_3_result.npz')
    M = result['M']
    K = result['K']
    WV = result['WV']
    #WV_hist = result['WV_hist']
    #Err_train = result['Err_train']
    #Err_test = result['Err_test']

    xn = 15  # 등고선 표시 해상도
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    xx = (xx0, xx1)

    x = np.c_[np.reshape(xx0, xn * xn), np.reshape(xx1, xn * xn)]
    y, a, z, b = FFNN(WV, M, K, x)

    print(WV)

    fig = plt.figure(1, figsize=(12, 9))
    plt.subplots_adjust(left=0.075, bottom=0.05, right=0.95,
                        top=0.95, wspace=0.4, hspace=0.4)

    for m in range(M):
        ax = fig.add_subplot(3, 4, 1 + m * 4, projection='3d')
        show_activation3d(ax, b[:, m], xn, xx, [-10, 10], '$b_{0:d}$'.format(m))
        ax = fig.add_subplot(3, 4, 2 + m * 4, projection='3d')
        show_activation3d(ax, z[:, m], xn, xx, [0, 1], '$z_{0:d}$'.format(m))
    for k in range(K):
        ax = fig.add_subplot(3, 4, 3 + k * 4, projection='3d')
        show_activation3d(ax, a[:, k], xn, xx, [-5, 5], '$a_{0:d}$'.format(k))
        ax = fig.add_subplot(3, 4, 4 + k * 4, projection='3d')
        show_activation3d(ax, y[:, k], xn, xx, [0, 1], '$y_{0:d}$'.format(k))

    plt.show()

