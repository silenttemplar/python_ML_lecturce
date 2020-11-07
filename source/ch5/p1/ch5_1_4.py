import numpy as np
import matplotlib.pyplot as plt
from source.ch5.p1.ch5_1_2 import mse_line
from source.ch5.p1.ch5_1_3 import fit_line_num

# line 생성
def show_line(w, x):
    xb = np.linspace(x[0], x[1], 100)
    y = w[0] * xb + w[1]
    plt.plot(xb, y, color=(.5, .5, .5), linewidth=4)

if __name__ == '__main__':
    #  ch5_1_1.py 저장한 data 로드
    data = np.load('ch5_1_data.npz')
    X = data['X']
    T = data['T']
    X_min = data['X_min']
    X_max = data['X_max']

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

    W0, W1, dMSE, W_history = fit_line_num(X, T)
    print('반복횟수 {0}'.format(W_history.shape[0]))
    print('W=[{0:.6f}, {1:.6f}]'.format(W0, W1))
    print('dMSE=[{0:.6f}, {1:6f}]'.format(dMSE[0], dMSE[1]))
    print('MSE={0:6f}'.format(mse_line(X, T, [W0, W1])))

    W = np.array([W0, W1])
    mse = mse_line(X, T, W)

    print("w0={0:.3f}, w1={1:.3f}".format(W0, W1))
    print("SD={0:.3f}cm".format(np.sqrt(mse)))

    plt.figure(figsize=(4, 4))
    show_line(W, (X_min, X_max))
    plt.plot(X, T, marker='o', linestyle='None', color='cornflowerblue', markeredgecolor='black')
    plt.xlim(X_min, X_max)
    plt.grid(True)
    plt.show()

