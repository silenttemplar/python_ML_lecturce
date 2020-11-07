import numpy as np
import matplotlib.pyplot as plt

def logistic(x, w):
    y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
    return y

def show_logistic(w, X):
    xb = np.linspace(X[0], X[1], 100)
    y = logistic(xb, w)
    plt.plot(xb, y, color='gray', linewidth=4)

    i = np.min(np.where(y > 0.5))
    B = (xb[i - 1] + xb[i]) / 2
    return B

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_2_data.npz')

    X_min = data['X_min']
    X_max = data['X_max']
    X_n = data['X_n']
    X_col = data['X_col']

    X = data['X']
    T = data['T']

    W = [8, -10]

    plt.figure(figsize=(3, 3))
    B = show_logistic(W, (X_min, X_max))
    plt.plot([B, B], [-.5, 1.5], color='k', linestyle='--')
    plt.grid(True)
    plt.show()

