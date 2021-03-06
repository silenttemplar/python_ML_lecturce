import numpy as np
import time
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from source.ch8.p3.ch8_3_1 import Get_data

np.random.seed(1)

def Create_model_CNN():
    model = Sequential()

    # 3x3 필터를 8개 사용.
    model.add(Conv2D(8, (3, 3), padding='same',
                     input_shape=(28, 28, 1),
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = Get_data()
    model = Create_model_CNN()

    startTime = time.time()
    history = model.fit(x_train, y_train, epochs=10, batch_size=1000,
                        verbose=1, validation_data=(x_test, y_test))
    calculation_time = time.time() - startTime

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Calculation time:{0:.3f} sec".format(calculation_time))
