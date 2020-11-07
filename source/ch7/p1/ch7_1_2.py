import numpy as np
import matplotlib.pyplot as plt

def show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1],
                 linestyle='none', marker='o', markeredgecolor='black')

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch7_1_data.npz')

    X_train = data['X_train']
    X_test = data['X_test']
    T_train = data['T_train']
    T_test = data['T_test']

    X_range0 = data['X_range0']
    X_range1 = data['X_range1']

    plt.figure(1, figsize=(8, 3.7))
    plt.subplot(1, 2, 1)
    show_data(X_train, T_train)
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.title('Training Data')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    show_data(X_test, T_test)
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.title('Test Data')
    plt.grid(True)
    plt.show()
