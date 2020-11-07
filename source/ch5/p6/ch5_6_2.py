import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def model(x, w):
    y = w[0] - w[1] * np.exp(-w[2] * x)
    return y

def show_model(w, X):
    xb = np.linspace(X[0], X[1], 100)
    y = model(xb, w)
    plt.plot(xb, y, c=[.5, .5, .5], lw=4)

def mse_model(w, x, t):
    y = model(x, w)
    mse = np.mean((y - t )**2)
    return mse

def fit_model(w_init, x, t):
    res1 = minimize(mse_model, w_init, args=(x,t), method='powell')
    return res1.x


if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch5_6_data.npz')

    X = data['X']
    T = data['T']
    X_min = data['X_min']
    X_max = data['X_max']
    X_n = data['X_n']

    W_init = [100, 0, 0]
    W = fit_model(W_init, X, T)
    mse = mse_model(W, X, T)

    print("w0={0:.1f}, w1={1:.1f}, w2={2:.1f}".format(W[0], W[1], W[2]))
    print("SD={0:.2f}cm".format(np.sqrt(mse)))

    plt.figure(figsize=(4, 4))
    show_model(W, (X_min, X_max))
    plt.plot(X, T, marker='o', linestyle='none',
             color='cornflowerblue', markeredgecolor='black')
    plt.xlim(X_min, X_max)
    plt.grid(True)
    plt.show()
