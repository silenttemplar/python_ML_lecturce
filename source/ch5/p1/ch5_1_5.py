import numpy as np
import matplotlib.pyplot as plt
from source.ch5.p1.ch5_1_2 import mse_line
from source.ch5.p1.ch5_1_4 import show_line

# 해석적 해 구하기
def fit_line(x, t):
    mx = np.mean(x)
    mt = np.mean(t)
    mtx = np.mean(t * x)
    mxx = np.mean(x * x)

    w0 = (mtx - mt * mx) / (mxx - mx**2)
    w1 = mt - w0 * mx
    return np.array([w0, w1])

if __name__ == '__main__':
    #  ch5_1_1.py 저장한 data 로드
    data = np.load('ch5_1_data.npz')
    X = data['X']
    T = data['T']
    X_min = data['X_min']
    X_max = data['X_max']

    W = fit_line(X, T)
    mse = mse_line(X, T, W)

    print("w0={0:.3f}, w1={1:.3f}".format(W[0], W[1]))
    print("SD={0:.3f}cm".format(np.sqrt(mse)))

    plt.figure(figsize=(4, 4))
    show_line(W, (X_min, X_max))
    plt.plot(X, T, marker='o', linestyle='None', color='cornflowerblue', markeredgecolor='black')
    plt.xlim(X_min, X_max)
    plt.grid(True)
    plt.show()
