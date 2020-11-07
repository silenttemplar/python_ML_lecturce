import matplotlib.pyplot as plt
from keras.datasets import mnist

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    plt.figure(1, figsize=(12, 3.2))
    plt.subplots_adjust(wspace=0.5)
    plt.gray()

    for id in range(3):
        plt.subplot(1, 3, id + 1)
        img = x_train[id, :, :]
        plt.pcolor(255 - img)
        plt.text(24.5, 26, "%d" % y_train[id], color='cornflowerblue', fontsize=18)
        plt.xlim(0, 27)
        plt.ylim(27, 0)
        plt.grid('on', color='white')
    plt.show()