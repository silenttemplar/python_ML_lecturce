import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu, s):
    return np.exp(-(x - mu)**2 / (2 * s**2))

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch5_3_data.npz')

    X = data['X']
    T = data['T']
    X_min = data['X_min']
    X_max = data['X_max']
    X_n = data['X_n']

    M = 4
    mu = np.linspace(5, 30, M)
    s = mu[1] - mu[0]
    xb = np.linspace(X_min, X_max, 100)

    plt.figure(figsize=(4,4))
    for j in range(M):
        y = gauss(xb, mu[j], s)
        plt.plot(xb, y, color='gray', linewidth=3)

    plt.grid(True)
    plt.xlim(X_min, X_max)
    plt.ylim(0, 1.2)
    plt.show()
