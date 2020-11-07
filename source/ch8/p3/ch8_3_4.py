import numpy as np
from source.ch8.p3.ch8_3_1 import Get_data
from source.ch8.p3.ch8_3_2 import Create_model_CNN

np.random.seed(1)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = Get_data()
    model = Create_model_CNN()
    history = model.fit(x_train, y_train, epochs=10, batch_size=1000,
                        verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)