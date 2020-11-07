import numpy as np
import matplotlib.pyplot as plt
from source.ch8.p3.ch8_3_1 import Get_data, process_filter
from source.ch8.p3.ch8_3_2 import Create_model_CNN

np.random.seed(1)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = Get_data()
    model = Create_model_CNN()
    history = model.fit(x_train, y_train, epochs=10, batch_size=1000,
                        verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    id_img = 12
    x_img = x_test[id_img, :, :, 0]
    img_h = 28
    img_w = 28
    x_img = x_img.reshape(img_h, img_w)
    
    # 원본이미지
    plt.figure(1, figsize=(12, 2.5))
    plt.gray()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.subplot(2, 9, 10)
    plt.pcolor(-x_img)
    plt.xlim(0, img_h)
    plt.ylim(img_w, 0)
    plt.xticks([], "")
    plt.yticks([], "")
    plt.title("Original")

    # 학습으로 생성한 필터
    w = model.layers[0].get_weights()[0]  # (A)
    max_w = np.max(w)
    min_w = np.min(w)
    for i in range(8):
        plt.subplot(2, 9, i + 2)
        w1 = w[:, :, 0, i]
        w1 = w1.reshape(3, 3)
        plt.pcolor(-w1, vmin=min_w, vmax=max_w)
        plt.xlim(0, 3)
        plt.ylim(3, 0)
        plt.xticks([], "")
        plt.yticks([], "")
        plt.title("%d" % i)
        plt.subplot(2, 9, i + 11)
        out_img = process_filter(x_img, w1.reshape(-1))
        plt.pcolor(-out_img)
        plt.xlim(0, img_w)
        plt.ylim(img_h, 0)
        plt.xticks([], "")
        plt.yticks([], "")
    plt.show()