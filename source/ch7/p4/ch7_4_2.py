import numpy as np
import matplotlib.pyplot as plt
from source.ch7.p4.ch7_4_1 import Create_model, Show_data

if __name__ == '__main__':
    data = np.load('../p1/ch7_1_data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    T_train = data['T_train']
    T_test = data['T_test']
    X_range0 = data['X_range0']
    X_range1 = data['X_range1']

    # 학습
    model = Create_model();
    history = model.fit(X_train, T_train, epochs=1000, batch_size=100,
                        verbose=0, validation_data=(X_test, T_test))
    score = model.evaluate(X_test, T_test, verbose=0)
    #print(history)

    # 리스트 7-2-(4)
    plt.figure(1, figsize=(12, 3))
    plt.subplots_adjust(wspace=0.5)

    # 학습 곡선 표시 --------------------------
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], 'black', label='training')  # (A)
    plt.plot(history.history['val_loss'], 'cornflowerblue', label='test')  # (B)
    plt.legend()

    # 정확도 표시 --------------------------
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], 'black', label='training')  # (C)
    plt.plot(history.history['val_accuracy'], 'cornflowerblue', label='test')  # (D)
    plt.legend()

    # 경계선 표시 --------------------------
    plt.subplot(1, 3, 3)
    Show_data(X_test, T_test)
    xn = 60  # 등고선 표시 해상도
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    x = np.c_[np.reshape(xx0, xn * xn), np.reshape(xx1, xn * xn)]
    y = model.predict(x)  # (E)
    K = 3
    for ic in range(K):
        f = y[:, ic]
        f = f.reshape(xn, xn)
        f = f.T
        cont = plt.contour(xx0, xx1, f, levels=[0.5, 0.9], colors=[
            'cornflowerblue', 'black'])
        cont.clabel(fmt='%1.1f', fontsize=9)
        plt.xlim(X_range0)
        plt.ylim(X_range1)
    plt.show()