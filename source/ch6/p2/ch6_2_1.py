import numpy as np

np.random.seed(seed=0)
X_min = 0
X_max = 2.5
X_n = 30
X_col = ['cornflowerblue', 'gray']

X = np.zeros(X_n)
T = np.zeros(X_n, dtype=np.uint8)

Dist_s = [0.4, 0.8]
Dist_w = [0.8, 1.6]
Pi = 0.5

for n in range(X_n):
    wk = np.random.rand()
    T[n] = 0 * (wk < Pi) + 1 * (wk >= Pi)
    X[n] = np.random.rand() * Dist_w[T[n]] + Dist_s[T[n]]

if __name__ == '__main__':
    # data 저장
    np.savez('ch6_2_data.npz', X_min=X_min, X_max=X_max, X_n=X_n, X_col=X_col, X=X, T=T)

    print('X={0:s}'.format(str(np.round(X, 2))))
    print('T={0:s}'.format(str(T)))