import numpy as np
import matplotlib.pyplot as plt
from source.ch6.p3.ch6_3_3 import logistic

def show_contour_logistic(w, X_range):
    xn = 30 # 파라미터의 분할 수
    x0 = np.linspace(X_range[0][0], X_range[0][1], xn)
    x1 = np.linspace(X_range[1][0], X_range[1][1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    y = logistic(xx0, xx1, w)
    cont = plt.contour(xx0, xx1, y, levels=(0.2, 0.5, 0.8),
                       colors=['k', 'cornflowerblue', 'k'])
    cont.clabel(fmt='%1.1f', fontsize=10)
    plt.grid(True)

if __name__ == '__main__':
    # 저장한 data 로드
    data = np.load('ch6_3_data.npz')

    X = data['X']
    T2 = data['T2']

    X_range0 = [-3, 3]
    X_range1 = [-3, 3]

    W = [-1, -1, -1]

    plt.figure(figsize=(3, 3))
    show_contour_logistic(W, X_range=(X_range0, X_range1))
    plt.show()
