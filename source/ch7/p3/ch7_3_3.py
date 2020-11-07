import numpy as np
import matplotlib.pyplot as plt
from source.ch7.p1.ch7_1_2 import show_data
from source.ch7.p2.ch7_2_6 import show_FFNN

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('../p2/ch7_2_data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    T_train = data['T_train']
    T_test = data['T_test']
    X_range0 = data['X_range0']
    X_range1 = data['X_range1']

    # 계산결과
    result = np.load('ch7_3_result.npz')
    M = result['M']
    K = result['K']
    WV = result['WV']
    WV_hist = result['WV_hist']
    Err_train = result['Err_train']
    Err_test = result['Err_test']

    plt.figure(1, figsize=(12, 3))
    plt.subplots_adjust(wspace=0.5)

    plt.subplot(1, 3, 1)
    plt.plot(Err_train, 'black', label="training")
    plt.plot(Err_test, 'cornflowerblue', label='test')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(WV_hist[:, :M * 3], 'black')
    plt.plot(WV_hist[:, M * 3:], 'cornflowerblue')

    plt.subplot(1, 3, 3)
    show_data(X_test, T_test)
    show_FFNN(WV, M, K, X_range=(X_range0, X_range1))
    plt.show()