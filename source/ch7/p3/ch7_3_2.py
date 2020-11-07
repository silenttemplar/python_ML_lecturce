import numpy as np
import time
from source.ch7.p2.ch7_2_2 import CE_FFNN
from source.ch7.p3.ch7_3_1 import dCE_FFNN

np.random.seed(1)

# 해석적 미분을 사용한 경사 하강법
def Fit_FNN(wv_init, M, K, x_train, t_train, x_test, t_test, n, alpha):
    wv = wv_init.copy()
    err_train = np.zeros(n)
    err_test = np.zeros(n)
    wv_hist = np.zeros((n, len(wv_init)))
    epsilon = 0.001

    for i in range(n):
        wv = wv - alpha * dCE_FFNN(wv, M, K, x_train, t_train) # (A)
        err_train[i] = CE_FFNN(wv, M, K, x_train, t_train)
        err_test[i] = CE_FFNN(wv, M, K, x_test, t_test)
        wv_hist[i, :] = wv
    return wv, wv_hist, err_train, err_test

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../p2/ch7_2_data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    T_train = data['T_train']
    T_test = data['T_test']

    #X_range0 = data['X_range0']
    #X_range1 = data['X_range1']

    startTime = time.time()

    M = 2
    K = 3
    WV_init = np.random.normal(0, 0.01, M * 3 + K * (M + 1))
    N_step = 1000
    alpha = 1

    WV, WV_hist, Err_train, Err_test = Fit_FNN( WV_init, M, K, X_train, T_train, X_test, T_test, N_step, alpha)

    calculation_time = time.time() - startTime
    print("Calculation time:{0:.3f} sec".format(calculation_time))

    # data 저장
    np.savez('ch7_3_result.npz', WV_hist=WV_hist, M=M, K=K, N_step=N_step, alpha=alpha, WV=WV, Err_train=Err_train, Err_test=Err_test)


