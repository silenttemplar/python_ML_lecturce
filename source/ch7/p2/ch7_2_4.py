import numpy as np
import time
from source.ch7.p2.ch7_2_3 import dCE_FFNN_num
from source.ch7.p2.ch7_2_2 import CE_FFNN

np.random.seed(1)

def Fit_FFNN_num(wv_init, M, K, x_train, t_train, x_test, t_test, n, alpha):
    wvt = wv_init

    err_train = np.zeros(n)
    err_test = np.zeros(n)
    wv_hist = np.zeros((n, len(wv_init)))
    #epsilon = 0.001
    for i in range(n):
        wvt = wvt - alpha * dCE_FFNN_num(wvt, M, K, x_train, t_train)
        err_train[i] = CE_FFNN(wvt, M, K, x_train, t_train)
        err_test[i] = CE_FFNN(wvt, M, K, x_test, t_test)
        wv_hist[i, :] = wvt
    return wvt, wv_hist, err_train, err_test

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch7_2_data.npz')

    X_train = data['X_train']
    X_test = data['X_test']
    T_train = data['T_train']
    T_test = data['T_test']

    startTime = time.time()

    M = 2
    K = 3
    WV_init = np.random.normal(0, 0.01, M * 3 + K * (M + 1))
    N_step = 1000  # (B) 학습 단계
    alpha = 0.5
    WV, WV_hist, Err_train, Err_test = Fit_FFNN_num(
        WV_init, M, K, X_train, T_train, X_test, T_test, N_step, alpha)
    calculation_time = time.time() - startTime
    print("Calculation time:{0:.3f} sec".format(calculation_time))

    # data 저장
    np.savez('ch7_2_result.npz',
             M=M, K=K,
             WV=WV, WV_hist=WV_hist,
             Err_train=Err_train, Err_test=Err_test)