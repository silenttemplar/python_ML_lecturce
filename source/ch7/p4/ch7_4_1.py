import numpy as np
import matplotlib.pyplot as plt
import time
import keras.optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation

np.random.seed(1)

def Show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1],
                 linestyle='none', marker='o',
                 markeredgecolor='black',
                 color=c[i], alpha=0.8)
    plt.grid(True)

def Create_model():
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='sigmoid', kernel_initializer='uniform'))
    model.add(Dense(3, activation='softmax', kernel_initializer='uniform'))
    sgd = keras.optimizers.SGD(lr=1, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    data = np.load('../p1/ch7_1_data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    T_train = data['T_train']
    T_test = data['T_test']
    X_range0 = data['X_range0']
    X_range1 = data['X_range1']

    model = Create_model()

    startTime = time.time()
    history = model.fit(X_train, T_train, epochs=1000, batch_size=100,
                        verbose=0, validation_data=(X_test, T_test))
    score = model.evaluate(X_test, T_test, verbose=0)
    print('cross entropy {0:.3f}, accuracy {1:3.2f}'.format(score[0], score[1]))

    calculation_time = time.time() - startTime
    print("Calculation time:{0:.3f} sec".format(calculation_time))