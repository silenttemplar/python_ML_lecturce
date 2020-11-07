import numpy as np
from source.ch7.p1.ch7_1_3 import FFNN

# 평균 교차 엔트로피 오차
def CE_FFNN(wv, M, K, x, t):
    N, D = x.shape
    y, a, z, b = FFNN(wv, M, K, x)
    ce = - np.dot(np.log(y.reshape(-1)), t.reshape(-1)) / N
    return ce

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch7_2_data.npz')

    X_train = data['X_train']
    X_test = data['X_test']
    T_train = data['T_train']
    T_test = data['T_test']

    WV = np.ones(15)
    M = 2
    K = 3
    print(CE_FFNN(WV, M, K, X_train[:2, :], T_train[:2, :]))