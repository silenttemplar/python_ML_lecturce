import numpy as np
import matplotlib.pyplot as plt
from source.ch8.p1.ch8_1_2 import Get_data
from source.ch8.p2.ch8_2_1 import Create_model_relu
from source.ch8.p1.ch8_1_4 import show_prediction

np.random.seed(1)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = Get_data()

    model = Create_model_relu()
    history = model.fit(x_train, y_train, epochs=10, batch_size=1000,
                        verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    show_prediction(model, x_test, y_test)
    plt.show()