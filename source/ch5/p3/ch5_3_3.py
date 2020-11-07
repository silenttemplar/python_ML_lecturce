import numpy as np
import matplotlib.pyplot as plt
from source.ch5.p3.ch5_3_2 import gauss

# 선형 기저함수 모델
def gauss_func(w, x):
    m = len(w) - 1
    mu = np.linspace(5, 30, m)
    s = mu[1] - mu[0]
    y = np.zeros_like(x)
    for j in range(m):
        y = y + w[j] * gauss(x, mu[j], s)
    y = y + w[m]
    return y

# 선형 기저함수 모델 MES
def mse_gauss_func(x, t, w):
    y = gauss_func(w, x)
    mse = np.mean((y - t)**2)
    return mse

# 선형 기저함수 해석적 해
def fit_gauss_func(x, t, m):
    mu = np.linspace(5, 30, m)
    s = mu[1] - mu[0]
    n = x.shape[0]
    psi = np.ones((n, m+1))
    for j in range(m):
        psi[:, j] = gauss(x, mu[j], s)
    psi_T = np.transpose(psi)

    b = np.linalg.inv(psi_T.dot(psi))
    c = b.dot(psi_T)
    w = c.dot(t)
    return w

# 가우스 기저함수 표시
def show_gauss_func(w, X):
    xb = np.linspace(X[0], X[1], 100)
    y = gauss_func(w, xb)
    plt.plot(xb, y, c=[.5, .5, .5], lw=4)

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch5_3_data.npz')

    X = data['X']
    T = data['T']
    X_min = data['X_min']
    X_max = data['X_max']
    X_n = data['X_n']

    M = 4
    W = fit_gauss_func(X, T, M)
    mse = mse_gauss_func(X, T, W)
    print('W={0:s}'.format(str(np.round(W,1))))
    print("SD={0:.2f}cm".format(np.sqrt(mse)))

    plt.figure(figsize=(4, 4))
    show_gauss_func(W, (X_min, X_max))
    plt.plot(X, T, marker='o', linestyle='None',
            color='cornflowerblue', markeredgecolor='black')
    plt.xlim(X_min, X_max)
    plt.grid(True)
    plt.show()