import numpy as np

np.random.seed(seed=1)

if __name__ == '__main__':
    # 데이터 수, 분포 수
    N = 200
    K = 3

    T = np.zeros((N, 3), dtype=np.uint8)
    X = np.zeros((N, 2))

    X_range0 = [-3, 3]
    X_range1 = [-3, 3]

    # 분포의 중심, 분포의 분산, 각 분포에 대한 비율
    Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]])
    Sig = np.array([[.7, .7], [.8, .3], [.3, .8]])
    Pi = np.array([0.4, 0.8, 1])

    for n in range(N):
        wk = np.random.rand()
        for k in range(K):
            if wk < Pi[k]:
                T[n, k] = 1
                break
        for k in range(2):
            X[n, k] = (np.random.randn() * Sig[T[n, :] == 1, k]) + Mu[T[n, :] == 1, k]

    TestRatio = 0.5
    X_n_training = int(N * TestRatio)
    X_train = X[:X_n_training, :]
    X_test = X[X_n_training:, :]
    T_train = T[:X_n_training, :]
    T_test = T[X_n_training:, :]

    # data 저장
    np.savez('ch7_1_data.npz',
             X_train=X_train, X_test=X_test,
             T_train=T_train, T_test=T_test,
             X_range0=X_range0, X_range1=X_range1)