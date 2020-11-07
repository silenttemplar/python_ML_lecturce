import numpy as np

# 활성화 함수 - 시그모이드
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

# 피드 포워드 신경 네트워크
def FFNN(wv, M, K, x):
    N, D = x.shape # 입력 차원

    # 중간층 뉴런의 가중치
    w = wv[:M * (D + 1)]
    w = w.reshape(M, (D + 1))

    # 출력층 뉴런의 가중치
    v = wv[M * (D + 1):]
    v = v.reshape((K, M + 1))

    # 중간층 뉴런의 입력 총합, 출력
    b = np.zeros((N, M + 1))
    z = np.zeros((N, M + 1))

    # 출력층 뉴런의 입력 총합, 출력
    a = np.zeros((N, K))
    y = np.zeros((N, K))

    for n in range(N):
        # 중간층의 계산
        for m in range(M):
            b[n, m] = np.dot(w[m, :], np.r_[x[n, :], 1]) # (A)
            z[n, m] = sigmoid(b[n, m])
        # 출력층의 계산
        z[n, M] = 1 # 더미 뉴런
        wkz = 0
        for k in range(K):
            a[n, k] = np.dot(v[k, :], z[n, :])
            wkz = wkz + np.exp(a[n, k])
        for k in range(K):
            y[n, k] = np.exp(a[n, k]) / wkz
    return y, a, z, b


if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch7_1_data.npz')

    X_train = data['X_train']
    X_test = data['X_test']
    T_train = data['T_train']
    T_test = data['T_test']

    #X_range0 = data['X_range0']
    #X_range1 = data['X_range1']

    WV = np.ones(15)
    M = 2
    K = 3
    print(FFNN(WV, M, K, X_train[:2, :]))