import numpy as np
import matplotlib.pyplot as plt
from source.ch5.p1.ch5_1_2 import mse_line

# 평균제곱오차 기울기
def dmse_line(x, t, w):
    y = w[0]*x + w[1]
    d_w0 = 2 * np.mean((y-t) * x)
    d_w1 = 2 * np.mean(y-t)
    return d_w0, d_w1

#d_w =dmse_line(X, T, [10, 165])
#print(np.round(d_w, 1))

# 경사하강법(steepest descent method)
def fit_line_num(x, t):
    w_init = [10.0, 165.0]
    alpha = 0.001
    i_max = 100000
    eps = 0.1

    w_i = np.zeros([i_max, 2])
    w_i[0, :] = w_init

    for i in range(1, i_max):
        dmse = dmse_line(x, t, w_i[i-1])
        w_i[i, 0] = w_i[i - 1, 0] - alpha * dmse[0]
        w_i[i, 1] = w_i[i - 1, 1] - alpha * dmse[1]
        if max(np.absolute(dmse)) < eps:
            break

    w0 = w_i[i, 0]
    w1 = w_i[i, 1]
    w_i = w_i[:i, :]
    return w0, w1, dmse, w_i

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

    plt.figure(figsize=(4, 4))
    cont = plt.contour(xx0, xx1, J, 30, colors='black', levels=[100, 1000, 10000, 100000])
    cont.clabel(fmt='%1.0f', fontsize=8)
    plt.grid(True)
    plt.plot(W_history[:, 0], W_history[:, 1], '.-', color='gray', markeredgecolor='cornflowerblue')
    plt.show()




