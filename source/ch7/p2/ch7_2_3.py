import numpy as np
import matplotlib.pyplot as plt
from source.ch7.p2.ch7_2_2 import CE_FFNN

np.random.seed(1)

# 평균 교차 엔트로피 오차 미분
def dCE_FFNN_num(wv, M, K, x, t):
    epsilon = 0.001
    dwv = np.zeros_like(wv)
    for iwv in range(len(wv)):
        wv_modified = wv.copy()
        wv_modified[iwv] = wv[iwv] - epsilon
        mse1 = CE_FFNN(wv_modified, M, K, x, t)
        wv_modified[iwv] = wv[iwv] + epsilon
        mse2 = CE_FFNN(wv_modified, M, K, x, t)
        dwv[iwv] = (mse2 - mse1) / (2 * epsilon)
    return dwv

def Show_WV(wv, M):
    N = wv.shape[0]
    plt.bar(range(1, M * 3 + 1), wv[:M * 3], align="center", color='black')
    plt.bar(range(M * 3 + 1, N + 1), wv[M * 3:],
            align="center", color='cornflowerblue')
    plt.xticks(range(1, N + 1))
    plt.xlim(0, N + 1)

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch7_2_data.npz')

    X_train = data['X_train']
    X_test = data['X_test']
    T_train = data['T_train']
    T_test = data['T_test']

    M = 2
    K = 3

    nWV = M * 3 + K * (M + 1)   # 가중치
    WV = np.random.normal(0, 1, nWV)
    dWV = dCE_FFNN_num(WV, M, K, X_train[:2, :], T_train[:2, :])
    print(dWV)

    plt.figure(1, figsize=(5, 3))
    Show_WV(dWV, M)
    plt.show()