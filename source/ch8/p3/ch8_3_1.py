import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils

def Get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_classes = 10

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    x_test = x_test.astype('float32')
    x_train = x_train.astype('float32')

    x_test = x_test / 255
    x_train = x_train / 255

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

# 가로 강조 필터
def Get_myfil1():
    return np.array([[1, 1, 1], [1, 1, 1], [-2, -2, -2]], dtype=float)

# 세로 강조 필터
def Get_myfil2():
    return np.array([[-2, 1, 1], [-2, 1, 1], [-2, 1, 1]], dtype=float)

# filter 적용
def process_filter(target_img, myfil):
    img_h = 28
    img_w = 28

    target_img = target_img.reshape(img_h, img_w)
    out_img = np.zeros_like(target_img)
    for ih in range(img_h - 3):
        for iw in range(img_w - 3):
            # filter 적용할 이미지 추출
            img_part = target_img[ih:ih +3, iw:iw + 3]
            # 추출된 이미지 내  filter 적용
            out_img[ih + 1, iw + 1] = np.dot(img_part.reshape(-1), myfil.reshape(-1))
    return out_img

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = Get_data()

    id_img = 2
    x_img = x_train[id_img, :, :, 0]

    myfil1 = Get_myfil1()
    myfil2 = Get_myfil2()
    out_img1 = process_filter(x_img, myfil1)
    out_img2 = process_filter(x_img, myfil2)

    plt.figure(1, figsize=(12, 3.2))
    plt.subplots_adjust(wspace=0.5)
    plt.gray()
    plt.subplot(1, 3, 1)
    plt.pcolor(1 - x_img)
    plt.xlim(-1, 29)
    plt.ylim(29, -1)

    plt.subplot(1, 3, 2)
    plt.pcolor(-out_img1)
    plt.xlim(-1, 29)
    plt.ylim(29, -1)

    plt.subplot(1, 3, 3)
    plt.pcolor(-out_img2)
    plt.xlim(-1, 29)
    plt.ylim(29, -1)
    plt.show()