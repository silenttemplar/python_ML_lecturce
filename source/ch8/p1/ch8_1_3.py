import numpy as np
import matplotlib.pyplot as plt
from source.ch8.p1.ch8_1_2 import Get_data, Create_model

np.random.seed(1)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = Get_data()
    model = Create_model()
    history = model.fit(x_train, y_train, epochs=10, batch_size=1000,
                        verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)


    plt.figure(1, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.5)

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='training', color='black')
    plt.plot(history.history['val_loss'], label='test',
             color='cornflowerblue')
    plt.ylim(0, 10)
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='training', color='black')
    plt.plot(history.history['val_accuracy'], label='test', color='cornflowerblue')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()