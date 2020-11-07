import numpy as np
import matplotlib.pyplot as plt
from source.ch8.p1.ch8_1_2 import Get_data
from source.ch8.p2.ch8_2_1 import Create_model_relu

np.random.seed(1)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = Get_data()

    model = Create_model_relu()
    history = model.fit(x_train, y_train, epochs=10, batch_size=1000,
                        verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    w = model.layers[0].get_weights()[0]

    plt.figure(1, figsize=(12, 3))
    plt.gray()
    plt.subplots_adjust(wspace=0.35, hspace=0.5)
    for i in range(16):
        plt.subplot(2, 8, i+1)
        w1 = w[:, i]
        w1 = w1.reshape(28, 28)
        plt.pcolor(-w1)
        plt.xlim(0, 27)
        plt.ylim(27, 0)
        plt.xticks([], "")
        plt.yticks([], "")
        plt.title("%d" % i)
    plt.show()