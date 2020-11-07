import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from source.ch8.p3.ch8_3_1 import Get_data

np.random.seed(1)

def Create_model_dropout():
    num_classes = 10

    model = Sequential()
    model.add(Conv2D(16, (3, 3),
                     input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # (A)
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # (B)
    model.add(Dropout(0.25))  # (C)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))  # (D)
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = Get_data()
    model = Create_model_dropout()

    startTime = time.time()
    history = model.fit(x_train, y_train, epochs=20, batch_size=1000,
                        verbose=1, validation_data=(x_test, y_test))
    calculation_time = time.time() - startTime

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Calculation time:{0:.3f} sec".format(calculation_time))