import numpy as np

np.random.seed(seed=1)
# 데이터 수, 분포 수
N = 100
K = 3

X = np.zeros((N, 2))

# 2클래스 분류용
T2 = np.zeros((N, 2), dtype=np.uint8)
# 3클래스 분류용
T3 = np.zeros((N, 3), dtype=np.uint8)

# 분포의 중심, 분포의 분산, 각 분포에 대한 비율
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]])
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]])
Pi = np.array([0.4, 0.8, 1])

for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < Pi[k]:
            T3[n, k] = 1
            break
    for k in range(2):
        X[n, k] = (np.random.randn() * Sig[T3[n, :] == 1, k]) + Mu[T3[n, :] == 1, k]
T2[:, 0] = T3[:, 0]
T2[:, 1] = T3[:, 1] | T3[:, 2]

if __name__ == '__main__':
    # data 저장
    np.savez('ch6_4_data.npz', N=N, K=K, T3=T3, T2=T2, X=X)

    print(X[:5, :])
    print(T2[:5, :])
    print(T3[:5, :])

