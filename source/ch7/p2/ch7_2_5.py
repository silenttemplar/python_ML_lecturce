import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 저장한 data 로드
    result = np.load('ch7_2_result.npz')

    M = result['M']
    K = result['K']
    WV = result['WV']
    WV_hist = result['WV_hist']
    Err_train = result['Err_train']
    Err_test = result['Err_test']

    plt.figure(1, figsize=(8, 3.7))
    plt.subplot(1, 2, 1)
    plt.plot(Err_train, 'black', label='training')
    plt.plot(Err_test, 'cornflowerblue', label='test')
    plt.title('mse')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(WV_hist[:, :M * 3], 'black')
    plt.plot(WV_hist[:, M * 3:], 'cornflowerblue')
    plt.title('bias')
    plt.grid(True)
    plt.show()