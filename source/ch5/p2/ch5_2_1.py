import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=1)

X_min = 4
X_max = 30
X_n = 16
X = 5 + 25 * np.random.rand(X_n)

Prm_c = [170, 108, 0.2]
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n)

# 나이
X0 = X
X0_min = 5
X0_max = 30

# 몸무게
np.random.seed(seed=1)
X1 = 23 * (T / 100)**2 + 2 * np.random.randn(X_n)
X1_min = 40
X1_max = 75

print(np.round(X0, 2))
print(np.round(X1, 2))
print(np.round(T, 2))

if __name__ == '__main__':
    np.savez('ch5_2_data.npz', X_n=X_n, T=T, X0=X0, X0_min=X0_min, X0_max=X0_max, X1=X1, X1_min=X1_min, X1_max=X1_max)