import numpy as np
import matplotlib.pyplot as plt
from source.ch7.p1.ch7_1_3 import FFNN
from source.ch7.p2.ch7_2_3 import dCE_FFNN_num

np.random.seed(1)

# 해석적 미분
def dCE_FFNN(wv, M, K, x, t):
    N, D = x.shape

    # wv을 w와 v로 되돌림
    w = wv[:M * (D + 1)]
    w = w.reshape(M, (D + 1))
    v = wv[M * (D + 1):]
    v = v.reshape((K, M + 1))

    # ① x를 입력하여 y를 얻음
    y, a, z, b = FFNN(wv, M, K, x)

    # 출력 변수의 준비
    dwv = np.zeros_like(wv)
    dw = np.zeros((M, D + 1))
    dv = np.zeros((K, M + 1))
    delta1 = np.zeros(M) # 1층 오차
    delta2 = np.zeros(K) # 2층 오차(k = 0 부분은 사용하지 않음)

    for n in range(N): # (A)
        # ② 출력층의 오차를 구하기
        for k in range(K):
            delta2[k] = (y[n, k] - t[n, k])
        # ③ 중간층의 오차를 구하기
        for j in range(M):
            delta1[j] = z[n, j] * (1 - z[n, j]) * np.dot(v[:, j], delta2)
        # ④ v의 기울기 dv를 구하기
        for k in range(K):
            dv[k, :] = dv[k, :] + delta2[k] * z[n, :] / N
        # ④ w의 기울기 dw를 구하기
        for j in range(M):
            dw[j, :] = dw[j, :] + delta1[j] * np.r_[x[n, :], 1] / N
    # dw와 dv를 합체시킨 dwv로 만들기
    dwv = np.c_[dw.reshape((1, M * (D + 1))), dv.reshape((1, K * (M + 1)))]
    dwv = dwv.reshape(-1)
    return dwv

def Show_dWV(wv, M):
    N = wv.shape[0]
    plt.bar(range(1, M * 3 + 1), wv[:M * 3], align="center", color='black')
    plt.bar(range(M * 3 + 1, N + 1), wv[M * 3:], align="center", color='cornflowerblue')
    plt.xticks(range(1, N + 1))
    plt.xlim(0, N + 1)

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../p2/ch7_2_data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    T_train = data['T_train']
    T_test = data['T_test']

    X_range0 = data['X_range0']
    X_range1 = data['X_range1']

    M = 2
    K = 3
    N = 2
    nWV = M * 3 + K * (M + 1)
    WV = np.random.normal(0, 1, nWV)

    dWV_ana = dCE_FFNN(WV, M, K, X_train[:N, :], T_train[:N, :])
    dWV_num = dCE_FFNN_num(WV, M, K, X_train[:N, :], T_train[:N, :])

    plt.figure(1, figsize=(8, 3))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1, 2, 1)
    Show_dWV(dWV_ana, M)
    plt.title('analitical')

    plt.subplot(1, 2, 2)
    Show_dWV(dWV_num, M)
    plt.title('numerical')
    plt.show()