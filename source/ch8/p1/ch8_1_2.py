import numpy as np
import time
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed(1)

def Create_model():
    model = Sequential()
    model.add(Dense(16, input_dim=784, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model

def Get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_classes = 10

    x_train = x_train.reshape(60000, 784)
    x_train = x_train.astype('float32')
    x_train = x_train / 255
    y_train = np_utils.to_categorical(y_train, num_classes)

    x_test = x_test.reshape(10000, 784)
    x_test = x_test.astype('float32')
    x_test = x_test / 255
    y_test = np_utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = Get_data()

    startTime = time.time()

    model = Create_model()
    history = model.fit(x_train, y_train, epochs=10, batch_size=1000,
                        verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    calculation_time = time.time() - startTime
    print("Calculation time:{0:.3f} sec".format(calculation_time))