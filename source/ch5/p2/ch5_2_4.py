import numpy as np
import matplotlib.pyplot as plt
from source.ch5.p2.ch5_2_3 import show_plane, show_data, mse_place

# 해석적 해 구하기
def fit_plane(x0, x1, t):
    c_tx0 = np.mean(t * x0) - np.mean(t) * np.mean(x0)
    c_tx1 = np.mean(t * x1) - np.mean(t) * np.mean(x1)
    c_x0x1 = np.mean(x0 * x1) - np.mean(x0) * np.mean(x1)
    v_x0 = np.var(x0)
    v_x1 = np.var(x1)
    w0 = (c_tx1 * c_x0x1 - v_x1 * c_tx0) / (c_x0x1**2 - v_x0 * v_x1)
    w1 = (c_tx0 * c_x0x1 - v_x0 * c_tx1) / (c_x0x1**2 - v_x0 * v_x1)
    w2 = -w0 * np.mean(x0) - w1 * np.mean(x1) + np.mean(t)
    return np.array([w0, w1, w2])

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch5_2_data.npz')

    # 나이
    X0 = data['X0']
    X0_min = data['X0_min']
    X0_max = data['X0_max']
    # 몸무게
    X1 = data['X1']
    X1_min = data['X1_min']
    X1_max = data['X1_max']
    # 키
    T = data['T']

    W = fit_plane(X0, X1, T)
    mse = mse_place(X0, X1, T, W)
    print("w0={0:.1f}, w1={1:.1f}, w2={2:.1f}".format(W[0], W[1], W[2]))
    print("SD={0:.2f}cm".format(np.sqrt(mse)))

    plt.figure(figsize=(6, 5))
    ax = plt.subplot(1, 1, 1, projection='3d')
    show_plane(ax, W, (X0_min, X0_max), (X1_min, X1_max))
    show_data(ax, X0, X1, T)
    plt.show()
